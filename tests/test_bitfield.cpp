#include "bitfield.h"

#include <gtest/gtest.h>

int MyAdd(int a, int b)
{
    return a + b;
}




TEST(MyAdd, 4plus4)
{
    EXPECT_EQ(MyAdd(4, 4), 8);
    EXPECT_TRUE(true);
}//通过

TEST(MyAdd, 5plus5)
{
    EXPECT_EQ(MyAdd(5, 5), 10);
    EXPECT_TRUE(true);
}//不通过
TEST(MyAdd, 5plus7)
{
    EXPECT_EQ(MyAdd(5, 7), 8);
    EXPECT_TRUE(true);
}//不通过