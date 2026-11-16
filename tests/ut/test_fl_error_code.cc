// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#include "catch2/catch_test_macros.hpp"
#include "knowhere/segcore_error_code.h"

TEST_CASE("FL device errors map to permanent segcore errors", "[error_code][fl]") {
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::fl_device_error) ==
                   knowhere::StatusCategory::permanent_error);
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::fl_device_error) ==
                   milvus::ErrorCode::KnowhereError);
    REQUIRE_FALSE(knowhere::IsInputError(knowhere::Status::fl_device_error));
    REQUIRE_FALSE(knowhere::IsTransientError(knowhere::Status::fl_device_error));
}
