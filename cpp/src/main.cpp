#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <poll.h>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

struct SpeechRequest {
    std::string model;
    std::string input;
    std::string voice = "alloy";
    std::string response_format = "mp3";
    double speed = 1.0;
    std::optional<std::string> language;
};

struct WorkerReply {
    bool ok = false;
    int status = 500;
    std::string error_type = "server_error";
    std::string error_message = "unknown error";
    std::optional<std::string> error_param;
    std::optional<std::string> error_code;

    std::string content_type = "application/octet-stream";
    std::string language;
    std::string speaker;
    std::vector<unsigned char> audio;
};

struct Options {
    bool serve = false;
    std::string host = "0.0.0.0";
    int port = 8000;

    std::string onnx_dir = "onnx_models";
    std::string onnx;
    std::string default_language = "ZH_MIX_EN";

    std::string torch_device = "auto";
    std::string tts_device = "cuda";
    std::string bert_device = "cuda";
    int workers = 4;

    std::string python_bin;

    std::string text;
    std::string output = "speech.wav";
    std::string voice = "ZH_MIX_EN";
    std::string response_format = "wav";
    double speed = 1.0;
    std::optional<std::string> language;
};

static std::string make_openai_error_json(
    const std::string& message,
    const std::string& type,
    const std::optional<std::string>& param,
    const std::optional<std::string>& code) {
    rapidjson::Document doc;
    doc.SetObject();
    auto& alloc = doc.GetAllocator();

    rapidjson::Value error(rapidjson::kObjectType);
    error.AddMember("message", rapidjson::Value(message.c_str(), alloc), alloc);
    error.AddMember("type", rapidjson::Value(type.c_str(), alloc), alloc);

    if (param.has_value()) {
        error.AddMember("param", rapidjson::Value(param->c_str(), alloc), alloc);
    } else {
        error.AddMember("param", rapidjson::Value().SetNull(), alloc);
    }

    if (code.has_value()) {
        error.AddMember("code", rapidjson::Value(code->c_str(), alloc), alloc);
    } else {
        error.AddMember("code", rapidjson::Value().SetNull(), alloc);
    }

    doc.AddMember("error", error, alloc);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    return buffer.GetString();
}

static bool read_file_bytes(const fs::path& path, std::vector<unsigned char>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) {
        return false;
    }
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (size < 0) {
        return false;
    }
    out.resize(static_cast<size_t>(size));
    if (size > 0) {
        if (!ifs.read(reinterpret_cast<char*>(out.data()), size)) {
            return false;
        }
    }
    return true;
}

static bool write_all(int fd, const std::string& data) {
    size_t total = 0;
    while (total < data.size()) {
        ssize_t written = ::write(fd, data.data() + total, data.size() - total);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        total += static_cast<size_t>(written);
    }
    return true;
}

static bool read_line_timeout(int fd, std::string& line, int timeout_ms) {
    line.clear();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;
        }

        int remaining = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());

        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLIN;
        pfd.revents = 0;

        int pr = ::poll(&pfd, 1, remaining);
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (pr == 0) {
            return false;
        }

        if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            return false;
        }

        char c = 0;
        ssize_t n = ::read(fd, &c, 1);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }

        if (c == '\n') {
            return true;
        }
        line.push_back(c);

        if (line.size() > 4 * 1024 * 1024) {
            return false;
        }
    }
}

static fs::path make_temp_audio_path() {
    std::string templ = (fs::temp_directory_path() / "melo_cpp_audio_XXXXXX").string();
    std::vector<char> mutable_templ(templ.begin(), templ.end());
    mutable_templ.push_back('\0');
    int fd = ::mkstemp(mutable_templ.data());
    if (fd >= 0) {
        ::close(fd);
    }
    return fs::path(mutable_templ.data());
}

static std::string detect_python_bin(const Options& opts) {
    if (!opts.python_bin.empty()) {
        return opts.python_bin;
    }

    const char* env_python = std::getenv("MELO_CPP_PYTHON");
    if (env_python != nullptr && std::strlen(env_python) > 0) {
        return env_python;
    }

    fs::path local_venv = fs::current_path() / ".venv" / "bin" / "python";
    if (fs::exists(local_venv)) {
        return local_venv.string();
    }

    return "python3";
}

class WorkerProcess {
public:
    explicit WorkerProcess(Options options)
        : opts_(std::move(options)) {}

    ~WorkerProcess() {
        stop();
    }

    bool start(std::string& err) {
        if (started_) {
            return true;
        }

        int stdin_pipe[2] = {-1, -1};
        int stdout_pipe[2] = {-1, -1};

        if (::pipe(stdin_pipe) != 0) {
            err = "pipe(stdin) failed";
            return false;
        }
        if (::pipe(stdout_pipe) != 0) {
            ::close(stdin_pipe[0]);
            ::close(stdin_pipe[1]);
            err = "pipe(stdout) failed";
            return false;
        }

        std::string python_bin = detect_python_bin(opts_);
        std::vector<std::string> args = {
            python_bin,
            "-u",
            "-m",
            "melo.cpp_worker",
            "--default-language", opts_.default_language,
            "--onnx-dir", opts_.onnx_dir,
            "--torch-device", opts_.torch_device,
            "--tts-device", opts_.tts_device,
            "--bert-device", opts_.bert_device,
            "--workers", std::to_string(opts_.workers),
        };

        if (!opts_.onnx.empty()) {
            args.push_back("--onnx");
            args.push_back(opts_.onnx);
        }

        pid_t pid = ::fork();
        if (pid < 0) {
            ::close(stdin_pipe[0]);
            ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]);
            ::close(stdout_pipe[1]);
            err = "fork failed";
            return false;
        }

        if (pid == 0) {
            ::dup2(stdin_pipe[0], STDIN_FILENO);
            ::dup2(stdout_pipe[1], STDOUT_FILENO);

            ::close(stdin_pipe[0]);
            ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]);
            ::close(stdout_pipe[1]);

            std::vector<char*> argv;
            argv.reserve(args.size() + 1);
            for (auto& item : args) {
                argv.push_back(const_cast<char*>(item.c_str()));
            }
            argv.push_back(nullptr);

            ::execvp(argv[0], argv.data());
            std::fprintf(stderr, "Failed to exec worker: %s\n", std::strerror(errno));
            _exit(127);
        }

        pid_ = pid;
        write_fd_ = stdin_pipe[1];
        read_fd_ = stdout_pipe[0];

        ::close(stdin_pipe[0]);
        ::close(stdout_pipe[1]);

        started_ = true;
        return true;
    }

    void stop() {
        if (!started_) {
            return;
        }

        rapidjson::Document doc;
        doc.SetObject();
        auto& alloc = doc.GetAllocator();
        doc.AddMember("id", rapidjson::Value("shutdown", alloc), alloc);
        doc.AddMember("action", rapidjson::Value("quit", alloc), alloc);

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        doc.Accept(writer);
        std::string quit_line = std::string(buf.GetString()) + "\n";
        (void)write_all(write_fd_, quit_line);

        ::close(write_fd_);
        ::close(read_fd_);

        int status = 0;
        for (int i = 0; i < 20; ++i) {
            pid_t result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                started_ = false;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        ::kill(pid_, SIGTERM);
        ::waitpid(pid_, &status, 0);
        started_ = false;
    }

    WorkerReply synthesize(const SpeechRequest& req) {
        WorkerReply reply;
        if (!started_) {
            reply.error_message = "worker is not started";
            return reply;
        }

        fs::path tmp = make_temp_audio_path();
        std::string request_id = std::to_string(++request_id_);

        rapidjson::Document doc;
        doc.SetObject();
        auto& alloc = doc.GetAllocator();
        doc.AddMember("id", rapidjson::Value(request_id.c_str(), alloc), alloc);
        doc.AddMember("action", rapidjson::Value("synthesize", alloc), alloc);
        doc.AddMember("model", rapidjson::Value(req.model.c_str(), alloc), alloc);
        doc.AddMember("input", rapidjson::Value(req.input.c_str(), alloc), alloc);
        doc.AddMember("voice", rapidjson::Value(req.voice.c_str(), alloc), alloc);
        doc.AddMember("response_format", rapidjson::Value(req.response_format.c_str(), alloc), alloc);
        doc.AddMember("speed", req.speed, alloc);
        doc.AddMember("output_path", rapidjson::Value(tmp.string().c_str(), alloc), alloc);
        if (req.language.has_value()) {
            doc.AddMember("language", rapidjson::Value(req.language->c_str(), alloc), alloc);
        }

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        std::string line = std::string(buffer.GetString()) + "\n";

        if (!write_all(write_fd_, line)) {
            reply.error_message = "failed to send request to worker";
            if (fs::exists(tmp)) {
                fs::remove(tmp);
            }
            return reply;
        }

        std::string worker_line;
        bool found = false;
        for (int i = 0; i < 4096; ++i) {
            if (!read_line_timeout(read_fd_, worker_line, 180000)) {
                break;
            }

            rapidjson::Document resp;
            if (resp.Parse(worker_line.c_str()).HasParseError() || !resp.IsObject()) {
                continue;
            }

            if (!resp.HasMember("id") || !resp["id"].IsString()) {
                continue;
            }
            if (request_id != resp["id"].GetString()) {
                continue;
            }

            found = true;
            reply.ok = resp.HasMember("ok") && resp["ok"].IsBool() && resp["ok"].GetBool();

            if (reply.ok) {
                if (resp.HasMember("content_type") && resp["content_type"].IsString()) {
                    reply.content_type = resp["content_type"].GetString();
                }
                if (resp.HasMember("language") && resp["language"].IsString()) {
                    reply.language = resp["language"].GetString();
                }
                if (resp.HasMember("speaker") && resp["speaker"].IsString()) {
                    reply.speaker = resp["speaker"].GetString();
                }
                if (!read_file_bytes(tmp, reply.audio)) {
                    reply.ok = false;
                    reply.status = 500;
                    reply.error_type = "server_error";
                    reply.error_message = "failed to read synthesized audio";
                }
            } else {
                if (resp.HasMember("status") && resp["status"].IsInt()) {
                    reply.status = resp["status"].GetInt();
                } else {
                    reply.status = 500;
                }
                if (resp.HasMember("error_type") && resp["error_type"].IsString()) {
                    reply.error_type = resp["error_type"].GetString();
                }
                if (resp.HasMember("error_message") && resp["error_message"].IsString()) {
                    reply.error_message = resp["error_message"].GetString();
                }
                if (resp.HasMember("error_param") && resp["error_param"].IsString()) {
                    reply.error_param = resp["error_param"].GetString();
                }
                if (resp.HasMember("error_code") && resp["error_code"].IsString()) {
                    reply.error_code = resp["error_code"].GetString();
                }
            }
            break;
        }

        if (fs::exists(tmp)) {
            fs::remove(tmp);
        }

        if (!found) {
            reply.ok = false;
            reply.status = 500;
            reply.error_type = "server_error";
            reply.error_message = "worker response timeout or protocol error";
        }

        return reply;
    }

private:
    Options opts_;
    bool started_ = false;
    pid_t pid_ = -1;
    int write_fd_ = -1;
    int read_fd_ = -1;
    int64_t request_id_ = 0;
};

static void print_usage() {
    std::cout
        << "melo_cpp_zhmix_en usage:\n"
        << "  CLI mode:\n"
        << "    melo_cpp_zhmix_en --text \"你好，hello world\" --output out.wav [--onnx PATH]\n"
        << "  HTTP mode:\n"
        << "    melo_cpp_zhmix_en --serve --host 0.0.0.0 --port 8000 [--onnx PATH]\n"
        << "\n"
        << "Common options:\n"
        << "  --onnx-dir <dir>            Default ONNX root directory (default: onnx_models)\n"
        << "  --onnx <path>               Direct ONNX path (directory or onnx file parent)\n"
        << "  --default-language <lang>   Default language (default: ZH_MIX_EN)\n"
        << "  --tts-device <dev>          ONNX TTS provider hint (default: cuda)\n"
        << "  --bert-device <dev>         ONNX BERT provider hint (default: cuda)\n"
        << "  --torch-device <dev>        Torch device for fallback ops (default: auto)\n"
        << "  --workers <n>               Worker threads (default: 4)\n"
        << "  --python-bin <path>         Python binary path (default: .venv/bin/python or python3)\n";
}

static bool parse_args(int argc, char** argv, Options& opts, std::string& err) {
    std::unordered_map<std::string, std::string*> string_opts = {
        {"--host", &opts.host},
        {"--onnx-dir", &opts.onnx_dir},
        {"--onnx", &opts.onnx},
        {"--default-language", &opts.default_language},
        {"--torch-device", &opts.torch_device},
        {"--tts-device", &opts.tts_device},
        {"--bert-device", &opts.bert_device},
        {"--python-bin", &opts.python_bin},
        {"--text", &opts.text},
        {"--output", &opts.output},
        {"--voice", &opts.voice},
        {"--response-format", &opts.response_format},
        {"--language", nullptr},
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        }
        if (arg == "--serve") {
            opts.serve = true;
            continue;
        }

        if (arg == "--port") {
            if (i + 1 >= argc) {
                err = "--port expects a value";
                return false;
            }
            opts.port = std::stoi(argv[++i]);
            continue;
        }

        if (arg == "--workers") {
            if (i + 1 >= argc) {
                err = "--workers expects a value";
                return false;
            }
            opts.workers = std::stoi(argv[++i]);
            continue;
        }

        if (arg == "--speed") {
            if (i + 1 >= argc) {
                err = "--speed expects a value";
                return false;
            }
            opts.speed = std::stod(argv[++i]);
            continue;
        }

        auto it = string_opts.find(arg);
        if (it != string_opts.end()) {
            if (i + 1 >= argc) {
                err = arg + " expects a value";
                return false;
            }
            std::string value = argv[++i];
            if (arg == "--language") {
                opts.language = value;
            } else {
                *(it->second) = value;
            }
            continue;
        }

        err = "unknown argument: " + arg;
        return false;
    }

    return true;
}

static bool parse_speech_request_json(
    const std::string& body,
    SpeechRequest& out,
    std::string& err_msg,
    std::optional<std::string>& err_param) {
    rapidjson::Document doc;
    if (doc.Parse(body.c_str()).HasParseError() || !doc.IsObject()) {
        err_msg = "Invalid JSON body";
        return false;
    }

    if (!doc.HasMember("model") || !doc["model"].IsString()) {
        err_msg = "'model' is required";
        err_param = "model";
        return false;
    }

    if (!doc.HasMember("input") || !doc["input"].IsString()) {
        err_msg = "'input' is required";
        err_param = "input";
        return false;
    }

    out.model = doc["model"].GetString();
    out.input = doc["input"].GetString();

    if (doc.HasMember("voice") && doc["voice"].IsString()) {
        out.voice = doc["voice"].GetString();
    }
    if (doc.HasMember("response_format") && doc["response_format"].IsString()) {
        out.response_format = doc["response_format"].GetString();
    }
    if (doc.HasMember("speed")) {
        if (doc["speed"].IsNumber()) {
            out.speed = doc["speed"].GetDouble();
        } else {
            err_msg = "'speed' must be a number";
            err_param = "speed";
            return false;
        }
    }
    if (doc.HasMember("language") && doc["language"].IsString()) {
        out.language = std::string(doc["language"].GetString());
    }

    return true;
}

static bool check_api_key(
    const http::request<http::string_body>& req,
    const std::string& expected_key,
    int& status,
    std::string& err_json) {
    if (expected_key.empty()) {
        return true;
    }

    auto auth_it = req.find(http::field::authorization);
    if (auth_it == req.end()) {
        status = 401;
        err_json = make_openai_error_json(
            "Missing Bearer token",
            "authentication_error",
            std::nullopt,
            std::optional<std::string>("invalid_api_key"));
        return false;
    }

    auto auth_sv = auth_it->value();
    std::string auth(auth_sv.data(), auth_sv.size());
    constexpr const char* prefix = "Bearer ";
    if (auth.rfind(prefix, 0) != 0) {
        status = 401;
        err_json = make_openai_error_json(
            "Missing Bearer token",
            "authentication_error",
            std::nullopt,
            std::optional<std::string>("invalid_api_key"));
        return false;
    }

    std::string token = auth.substr(std::strlen(prefix));
    if (token != expected_key) {
        status = 401;
        err_json = make_openai_error_json(
            "Invalid API key",
            "authentication_error",
            std::nullopt,
            std::optional<std::string>("invalid_api_key"));
        return false;
    }

    return true;
}

static void handle_connection(tcp::socket socket, WorkerProcess& worker, const std::string& api_key) {
    beast::error_code ec;
    beast::flat_buffer buffer;
    http::request<http::string_body> req;
    http::read(socket, buffer, req, ec);
    if (ec) {
        return;
    }

    auto send_json = [&](http::status status, const std::string& body) {
        http::response<http::string_body> res{status, req.version()};
        res.set(http::field::content_type, "application/json");
        res.set(http::field::server, "melo_cpp_zhmix_en");
        res.keep_alive(false);
        res.body() = body;
        res.prepare_payload();
        http::write(socket, res, ec);
    };

    if (req.method() == http::verb::get && req.target() == "/healthz") {
        send_json(http::status::ok, "{\"status\":\"ok\"}");
        socket.shutdown(tcp::socket::shutdown_send, ec);
        return;
    }

    int auth_status = 0;
    std::string auth_err;
    if (!check_api_key(req, api_key, auth_status, auth_err)) {
        send_json(static_cast<http::status>(auth_status), auth_err);
        socket.shutdown(tcp::socket::shutdown_send, ec);
        return;
    }

    if (!(req.method() == http::verb::post && req.target() == "/v1/audio/speech")) {
        send_json(
            http::status::not_found,
            make_openai_error_json("Not found", "invalid_request_error", std::nullopt, std::nullopt));
        socket.shutdown(tcp::socket::shutdown_send, ec);
        return;
    }

    SpeechRequest sr;
    std::string parse_err;
    std::optional<std::string> parse_param;
    if (!parse_speech_request_json(req.body(), sr, parse_err, parse_param)) {
        send_json(
            http::status::bad_request,
            make_openai_error_json(parse_err, "invalid_request_error", parse_param, std::nullopt));
        socket.shutdown(tcp::socket::shutdown_send, ec);
        return;
    }

    WorkerReply wr = worker.synthesize(sr);
    if (!wr.ok) {
        send_json(
            static_cast<http::status>(wr.status),
            make_openai_error_json(wr.error_message, wr.error_type, wr.error_param, wr.error_code));
        socket.shutdown(tcp::socket::shutdown_send, ec);
        return;
    }

    http::response<http::vector_body<unsigned char>> audio_res{http::status::ok, req.version()};
    audio_res.set(http::field::content_type, wr.content_type);
    audio_res.set(http::field::server, "melo_cpp_zhmix_en");
    if (!wr.language.empty()) {
        audio_res.set("x-melo-language", wr.language);
    }
    if (!wr.speaker.empty()) {
        audio_res.set("x-melo-speaker", wr.speaker);
    }
    audio_res.keep_alive(false);
    audio_res.body() = std::move(wr.audio);
    audio_res.prepare_payload();
    http::write(socket, audio_res, ec);
    socket.shutdown(tcp::socket::shutdown_send, ec);
}

static int run_http_server(WorkerProcess& worker, const Options& opts) {
    asio::io_context ioc{1};
    tcp::endpoint endpoint{asio::ip::make_address(opts.host), static_cast<unsigned short>(opts.port)};
    tcp::acceptor acceptor{ioc};

    beast::error_code ec;
    acceptor.open(endpoint.protocol(), ec);
    if (ec) {
        std::cerr << "Failed to open acceptor: " << ec.message() << "\n";
        return 1;
    }

    acceptor.set_option(asio::socket_base::reuse_address(true), ec);
    acceptor.bind(endpoint, ec);
    if (ec) {
        std::cerr << "Failed to bind " << opts.host << ":" << opts.port << ": " << ec.message() << "\n";
        return 1;
    }

    acceptor.listen(asio::socket_base::max_listen_connections, ec);
    if (ec) {
        std::cerr << "Failed to listen: " << ec.message() << "\n";
        return 1;
    }

    const char* api_key = std::getenv("MELO_API_KEY");
    std::string expected_api_key = api_key ? api_key : "";

    std::cerr << "Server listening on http://" << opts.host << ":" << opts.port << "\n";
    while (true) {
        tcp::socket socket{ioc};
        acceptor.accept(socket, ec);
        if (ec) {
            std::cerr << "Accept error: " << ec.message() << "\n";
            continue;
        }
        handle_connection(std::move(socket), worker, expected_api_key);
    }
}

static int run_cli_once(WorkerProcess& worker, const Options& opts) {
    if (opts.text.empty()) {
        std::cerr << "--text is required in CLI mode\n";
        return 2;
    }

    SpeechRequest req;
    req.model = "melo-zhmix-en-cpp";
    req.input = opts.text;
    req.voice = opts.voice;
    req.response_format = opts.response_format;
    req.speed = opts.speed;
    req.language = opts.language;

    WorkerReply wr = worker.synthesize(req);
    if (!wr.ok) {
        std::cerr << "Synthesis failed: " << wr.error_message << "\n";
        return 3;
    }

    std::ofstream ofs(opts.output, std::ios::binary);
    if (!ofs.good()) {
        std::cerr << "Cannot open output file: " << opts.output << "\n";
        return 4;
    }
    if (!wr.audio.empty()) {
        ofs.write(reinterpret_cast<const char*>(wr.audio.data()), static_cast<std::streamsize>(wr.audio.size()));
    }

    std::cout << "Saved audio to " << opts.output << " (" << wr.audio.size() << " bytes, "
              << wr.content_type << ")\n";
    return 0;
}

int main(int argc, char** argv) {
    Options opts;
    std::string parse_err;
    if (!parse_args(argc, argv, opts, parse_err)) {
        std::cerr << "Argument error: " << parse_err << "\n";
        print_usage();
        return 2;
    }

    WorkerProcess worker(opts);
    std::string start_err;
    if (!worker.start(start_err)) {
        std::cerr << "Failed to start worker: " << start_err << "\n";
        return 1;
    }

    if (opts.serve) {
        return run_http_server(worker, opts);
    }
    return run_cli_once(worker, opts);
}
