// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
//
// OpenAI-compatible chat server built on cppgrad + Qwen3, with Hermes-style
// tool calling on both streaming and non-streaming paths.
//
// Usage:
//   ./chat_server --model-dir <path> [--config 27b] [--port 8080] [--quant]
//
// Endpoints:
//   GET  /                     - health check
//   GET  /v1/models            - list models
//   POST /v1/chat/completions  - chat completions (streaming + non-streaming)

#define CPPHTTPLIB_THREAD_POOL_COUNT 8
#include "cppgrad/server/chat_server.h"

#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "cppgrad/server/anthropic_messages.h"
#include "httplib.h" // header-only HTTP server (brew: cpp-httplib)

using cppgrad::server::ChatCompletionRequest;
using cppgrad::server::ChatServer;
namespace anthropic = cppgrad::server::anthropic;

struct ServerArgs {
    std::string model_dir;
    std::string config = "27b";
    std::string host = "0.0.0.0";
    int port = 8080;
    bool quant = true;
    // Speculative (MTP-style) decoding via an optional smaller draft model. Disabled unless
    // --draft-model is given (and --n-draft >= 2). The draft must share the main model's vocab.
    std::string draft_dir;
    std::string draft_config = "0.6b";
    int n_draft = 4;
    bool no_mtp = false; // disable MTP self-speculation even if the checkpoint provides it
};

static void print_usage(const char *prog_name) {
    std::cout << "Usage: " << prog_name << " --model-dir <path> [--config <name>] [--port 8080] [--quant]\n"
              << "       [--draft-model <path> [--draft-config 0.6b]] [--n-draft 4] [--no-mtp]\n"
              << "  --config: 0.6b, 1.5b, 3b, 4b, 7b, 27b\n"
              << "  speculative decode: auto-enabled if the checkpoint ships an MTP module;\n"
              << "  or pass --draft-model for draft-model speculation. --n-draft sets the window.\n";
}

static ServerArgs parse_args(int argc, char **argv) {
    ServerArgs a;
    enum { OPT_DRAFT_MODEL = 1000, OPT_DRAFT_CONFIG, OPT_N_DRAFT, OPT_NO_MTP };
    static struct option opts[] = {{"model-dir", required_argument, 0, 'm'},
                                   {"config", required_argument, 0, 'c'},
                                   {"port", required_argument, 0, 'p'},
                                   {"host", required_argument, 0, 'h'},
                                   {"quant", no_argument, 0, 'q'},
                                   {"draft-model", required_argument, 0, OPT_DRAFT_MODEL},
                                   {"draft-config", required_argument, 0, OPT_DRAFT_CONFIG},
                                   {"n-draft", required_argument, 0, OPT_N_DRAFT},
                                   {"no-mtp", no_argument, 0, OPT_NO_MTP},
                                   {"help", no_argument, 0, '?'},
                                   {0, 0, 0, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "m:c:p:h:q?", opts, nullptr)) != -1) {
        switch (c) {
        case 'm':
            a.model_dir = optarg;
            break;
        case 'c':
            a.config = optarg;
            break;
        case 'p':
            a.port = std::stoi(optarg);
            break;
        case 'h':
            a.host = optarg;
            break;
        case 'q':
            a.quant = true;
            break;
        case OPT_DRAFT_MODEL:
            a.draft_dir = optarg;
            break;
        case OPT_DRAFT_CONFIG:
            a.draft_config = optarg;
            break;
        case OPT_N_DRAFT:
            a.n_draft = std::stoi(optarg);
            break;
        case OPT_NO_MTP:
            a.no_mtp = true;
            break;
        case '?':
            print_usage(argv[0]);
            exit(0);
        }
    }

    if (a.model_dir.empty()) {
        std::cerr << "Error: The --model-dir (-m) parameter is mandatory.\n\n";
        print_usage(argv[0]);
        exit(1);
    }

    return a;
}

static void send_error(httplib::Response &res, int status, const std::string &msg) {
    nlohmann::json j = {{"error", {{"message", msg}, {"type", "invalid_request_error"}}}};
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

// Capture the raw request body to CPPGRAD_DUMP_REQUESTS (a file path) BEFORE any parsing/generation,
// fopen+fclose per call so it's flushed to disk even if the server then crashes. Diagnostic only.
static void dump_request(const char *endpoint, const std::string &body) {
    static const char *path = std::getenv("CPPGRAD_DUMP_REQUESTS");
    if (!path)
        return;
    if (FILE *f = std::fopen(path, "a")) {
        std::fprintf(f, "===== %s (%zu bytes) =====\n%s\n\n", endpoint, body.size(), body.c_str());
        std::fclose(f);
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0); // line-buffer stdout so startup/progress logs appear promptly
    auto args = parse_args(argc, argv);

    ChatServer chat;
    const bool dump_only = std::getenv("CPPGRAD_DUMP_ONLY") != nullptr;
    if (dump_only) {
        const char *p = std::getenv("CPPGRAD_DUMP_REQUESTS");
        std::cout << "[dump-only] model NOT loaded; capturing raw requests to "
                  << (p ? p : "(set CPPGRAD_DUMP_REQUESTS=<path>)") << " and returning 503\n";
    } else if (!chat.load_model(args.model_dir, args.config, args.quant)) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    if (dump_only) { /* no draft/mtp setup needed */
    } else if (!args.draft_dir.empty() && args.n_draft >= 2) {
        if (!chat.load_draft_model(args.draft_dir, args.draft_config, args.n_draft, args.quant))
            std::cerr << "Warning: failed to load draft model; speculative decoding disabled\n";
    } else {
        // MTP self-speculation is auto-enabled in load_model when present; honor --no-mtp / --n-draft.
        chat.set_mtp(!args.no_mtp, args.n_draft);
    }

    httplib::Server svr;
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Methods", "POST, GET, OPTIONS"},
                             {"Access-Control-Allow-Headers", "Content-Type, Authorization"}});

    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response &res) { res.status = 204; });

    svr.Get("/", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(R"({"status":"ok","server":"cppgrad-chat"})", "application/json");
    });

    svr.Get("/v1/models", [&chat](const httplib::Request &, httplib::Response &res) {
        res.set_content(nlohmann::json(chat.get_models()).dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&chat](const httplib::Request &req, httplib::Response &res) {
        dump_request("/v1/chat/completions", req.body);
        if (std::getenv("CPPGRAD_DUMP_ONLY"))
            return send_error(res, 503, "dump-only mode (request captured)");
        ChatCompletionRequest request;
        try {
            request = nlohmann::json::parse(req.body).get<ChatCompletionRequest>();
        } catch (const std::exception &e) {
            return send_error(res, 400, std::string("Invalid request: ") + e.what());
        }
        if (request.messages.empty())
            return send_error(res, 400, "messages must not be empty");

        if (request.stream.value_or(false)) {
            // Serialize generation: a single model is not safe to drive concurrently.
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream", [&chat, request](size_t /*offset*/, httplib::DataSink &sink) {
                    // Catch here: an uncaught throw in this worker thread would std::terminate the
                    // whole server (the response has already started, so we can only log + end).
                    try {
                        chat.chat_complete_stream(request, [&sink](const std::string &frame) {
                            return sink.write(frame.data(), frame.size());
                        });
                    } catch (const std::exception &e) {
                        std::cerr << "[stream] /v1/chat/completions generation error: " << e.what() << "\n";
                    }
                    sink.done();
                    return true;
                });
        } else {
            try {
                auto response = chat.chat_complete(request);
                res.set_content(nlohmann::json(response).dump(), "application/json");
            } catch (const std::exception &e) {
                return send_error(res, 500, std::string("Generation failed: ") + e.what());
            }
        }
    });

    // Anthropic Messages API (what Claude Code speaks): point ANTHROPIC_BASE_URL
    // at this server and it POSTs here.
    svr.Post("/v1/messages", [&chat](const httplib::Request &req, httplib::Response &res) {
        dump_request("/v1/messages", req.body);
        if (std::getenv("CPPGRAD_DUMP_ONLY"))
            return send_error(res, 503, "dump-only mode (request captured)");
        anthropic::Request request;
        try {
            request = anthropic::parse_request(nlohmann::json::parse(req.body));
        } catch (const std::exception &e) {
            return send_error(res, 400, std::string("Invalid request: ") + e.what());
        }
        if (request.messages.empty())
            return send_error(res, 400, "messages must not be empty");

        if (request.stream) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream", [&chat, request](size_t /*offset*/, httplib::DataSink &sink) {
                    try {
                        anthropic::handle_messages_stream(chat, request, [&sink](const std::string &frame) {
                            return sink.write(frame.data(), frame.size());
                        });
                    } catch (const std::exception &e) {
                        std::cerr << "[stream] /v1/messages generation error: " << e.what() << "\n";
                    }
                    sink.done();
                    return true;
                });
        } else {
            try {
                res.set_content(anthropic::handle_messages(chat, request).dump(), "application/json");
            } catch (const std::exception &e) {
                return send_error(res, 500, std::string("Generation failed: ") + e.what());
            }
        }
    });

    std::cout << "\n=== CppGrad Chat Server ===\n"
              << "Model:     " << chat.model_name() << "\n"
              << "Decode:    " << chat.decode_mode() << "\n"
              << "Listening: http://" << args.host << ":" << args.port << "\n"
              << "Endpoints: GET /  GET /v1/models\n"
              << "           POST /v1/chat/completions  (OpenAI)\n"
              << "           POST /v1/messages          (Anthropic / Claude Code)\n\n";

    if (!svr.listen(args.host.c_str(), args.port)) {
        std::cerr << "Failed to bind " << args.host << ":" << args.port << "\n";
        return 1;
    }
    return 0;
}
