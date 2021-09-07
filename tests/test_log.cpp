#include "log.h"

#include <gtest/gtest.h>

TEST(LOG, LOGType) {
	LOG(SILENT) << "log silent";
	LOG(WARNING) << "log warning";
	LOG(INFO) << "log info";
	LOG(DEBUG) << "log debug";
	LOG(IGNORE) << "log ignore";
}