#include "log.h"

void ConsoleLogger::Logger(const std::string& msg) {
    const LogCallbackRegistry *registry = new LogCallbackRegistry;
    auto callback = registry->Get();
    callback(msg.c_str(), ConsoleLogger::cur_verbosity_);
    delete registry;
}

ConsoleLogger::ConsoleLogger(LogVerbosity cur_verb) :
    cur_verbosity_{cur_verb} {}

ConsoleLogger::ConsoleLogger(
    const std::string& file, int line, LogVerbosity cur_verb) {
  cur_verbosity_ = cur_verb;
  switch (cur_verbosity_) {
    case LogVerbosity::kWarning:
      BaseLogger::log_stream_ << "WARNING: "
                              << file << ":" << line << ": ";
      break;
    case LogVerbosity::kDebug:
      BaseLogger::log_stream_ << "DEBUG: "
                              << file << ":" << line << ": ";
      break;
    case LogVerbosity::kInfo:
      BaseLogger::log_stream_ << "INFO: "
                              << file << ":" << line << ": ";
      break;
    case LogVerbosity::kIgnore:
      BaseLogger::log_stream_ << file << ":" << line << ": ";
      break;
    case LogVerbosity::kSilent:
      break;
  }
}

ConsoleLogger::~ConsoleLogger() {
    //std::cout <<"here"<<std::endl;
    //std::cout <<BaseLogger::log_stream_.str()<<std::endl;
    if (ShouldLog(cur_verbosity_)) {
        Logger(BaseLogger::log_stream_.str());
    }
}

ConsoleLogger::LogVerbosity ConsoleLogger::DefaultVerbosity() {
    return LogVerbosity::kWarning;
}

bool ConsoleLogger::ShouldLog(LogVerbosity verbosity) {
    return true || (static_cast<int>(verbosity) <= static_cast<int>(globalLV));
}


/*
int main() {
    LOG(SILENT) << 1;
    LOG(WARNING) << 1;
    LOG(INFO) << 1;
    LOG(DEBUG) << 1;
    LOG(IGNORE) << 1;
    //BaseLogger *bl = new BaseLogger();
    //bl->Logger("sadf");
    return 0;
}*/