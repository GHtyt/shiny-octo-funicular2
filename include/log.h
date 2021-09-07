#ifndef SOS_LOG_H_
#define SOS_LOG_H_

#include <sstream>
#include <iostream>
#include <string>

#ifndef LOG
#define LOG(severity) LOG_##severity.stream()

#define LOG_SILENT  ConsoleLogger(ConsoleLogger::LogVerbosity::kSilent)
#define LOG_WARNING ConsoleLogger(ConsoleLogger::LogVerbosity::kWarning)
#define LOG_INFO    ConsoleLogger(ConsoleLogger::LogVerbosity::kInfo)
#define LOG_DEBUG   ConsoleLogger(ConsoleLogger::LogVerbosity::kDebug)
#define LOG_IGNORE  ConsoleLogger(ConsoleLogger::LogVerbosity::kIgnore)

#endif


class BaseLogger {
    public:
        BaseLogger() {
            //log_stream_ << "[" << dmlc::DateLogger().HumanDate() << "] ";
        }
        std::ostream& stream() { return log_stream_; }  // NOLINT

    protected:
        std::ostringstream log_stream_;
};

class ConsoleLogger : public BaseLogger {
    
    public:
        enum class LogVerbosity {
            kSilent = 0,
            kWarning = 1,
            kInfo = 2,  
            kDebug = 3,  
            kIgnore = 4  
        };
        //using LV = LogVerbosity;

        const static LogVerbosity globalLV = LogVerbosity::kSilent;
    
    private:
        LogVerbosity cur_verbosity_;

    public:

        void Logger(const std::string& msg);
        //static void Configure(Args const& args);
        
        //static LogVerbosity GlobalVerbosity();
        static LogVerbosity DefaultVerbosity();
        static bool ShouldLog(LogVerbosity verbosity);

        ConsoleLogger() = delete;
        explicit ConsoleLogger(LogVerbosity cur_verb);
        ConsoleLogger(const std::string& file, int line, LogVerbosity cur_verb);
        ~ConsoleLogger();
    

};


class LogCallbackRegistry {
    public:
        using Callback = void (*)(const char*);
        using Callback_Verbosity = void (*)(const char*, ConsoleLogger::LogVerbosity);
        LogCallbackRegistry(): log_callback_([] (const char* msg, ConsoleLogger::LogVerbosity lv) { 
            switch (static_cast<int>(lv)) {
            case 0:
                std::cerr << "\033[3" << 2 << "m" << "[  SILENT  ]:   " << msg << "\033[0m" << std::endl;
                break;
            case 1:
                std::cerr << "\033[3" << 3 << "m" << "[  WARNING ]:   " << msg << "\033[0m" << std::endl;
                break;
            case 2:
                std::cerr << "\033[3" << 6 << "m" << "[  INFO    ]:   " << msg << "\033[0m" << std::endl;
                break;
            case 3:
                std::cerr << "\033[3" << 1 << "m" << "[  DEBUG   ]:   " << msg << "\033[0m" << std::endl;
                break;
            case 4:
                std::cerr << "\033[3" << 5 << "m" << "[  IGNORE  ]:   " << msg << "\033[0m" << std::endl;
                break;
            }}) {}
        inline void Register(Callback_Verbosity log_callback) {
            this->log_callback_ = log_callback;
        }
        inline Callback_Verbosity Get() const {
            return log_callback_;
        }
    private:
        Callback_Verbosity log_callback_;
};






#endif