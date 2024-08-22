/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <limits>

#include <executorch/extension/llm/custom_ops/op_sdpa.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::testing::TensorFactory;

exec_aten::Tensor op_scaled_dot_product_attention(
    const exec_aten::Tensor& query,
    const exec_aten::Tensor& key,
    const exec_aten::Tensor& value,
    const exec_aten::optional<exec_aten::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    exec_aten::optional<double> scale,
    exec_aten::Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::native::flash_attention_kernel_out(
      context, query, key, value, attn_mask, dropout_p, is_causal, scale, out);
}

/*
Most tests are generated by FACTO
*/

TEST(OpScaledDotProductAttentionTest, CorrectnessTest_105) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 1, 4, 4},
      {0.4320,
       0.1461,
       0.6817,
       0.8756,
       0.8619,
       0.9165,
       0.1050,
       0.0488,
       0.9832,
       0.8024,
       0.3185,
       0.7671,
       0.5988,
       0.2772,
       0.3965,
       0.1101});
  exec_aten::Tensor key = tfFloat.make(
      {1, 1, 4, 4},
      {0.4951,
       0.1630,
       0.7805,
       0.7971,
       0.7538,
       0.5109,
       0.0012,
       0.0018,
       0.3541,
       0.6563,
       0.5831,
       0.0022,
       0.7363,
       0.2270,
       0.1862,
       0.2762});
  exec_aten::Tensor value = tfFloat.make(
      {1, 1, 4, 4},
      {0.2914,
       0.4977,
       0.0895,
       0.3630,
       0.6552,
       0.1495,
       0.1673,
       0.5845,
       0.8988,
       0.6690,
       0.5082,
       0.9999,
       0.0609,
       0.7338,
       0.2203,
       0.6971});
  exec_aten::optional<exec_aten::Tensor> attn_mask;
  double dropout_p = 0;
  bool is_causal = false;
  exec_aten::optional<double> scale;
  exec_aten::Tensor ret_expected = tfFloat.make(
      {1, 1, 4, 4},
      {0.4473,
       0.5221,
       0.2302,
       0.6293,
       0.4910,
       0.5032,
       0.2501,
       0.6689,
       0.4630,
       0.5109,
       0.2368,
       0.6449,
       0.4741,
       0.5132,
       0.2444,
       0.6570});
  std::vector<int32_t> out_size = {1, 1, 4, 4};
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected, 1e-4, 1e-4);
}

TEST(OpScaledDotProductAttentionTest, CorrectnessTest_11) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 1, 1, 8},
      {75.25, -32.875, -96.375, 75.0, -5.25, -30.0, 71.5, -70.875});
  exec_aten::Tensor key = tfFloat.make(
      {1, 1, 1, 8},
      {50.125, 18.0, 72.625, -95.0, 47.25, -74.875, -46.375, -47.0});
  exec_aten::Tensor value = tfFloat.make(
      {1, 1, 1, 8},
      {99.375, 80.125, -81.0, 8.5, -70.375, -54.25, -80.25, 34.125});
  exec_aten::optional<exec_aten::Tensor> attn_mask =
      exec_aten::optional<exec_aten::Tensor>(
          tfFloat.full({1, 1}, std::numeric_limits<float>::infinity()));
  double dropout_p = 0.0;
  bool is_causal = false;
  exec_aten::optional<double> scale;
  std::vector<int32_t> out_size(query.sizes().begin(), query.sizes().end());
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  // Pytorch says these should be NAN
  // but output is -NAN. Both are really the same though
  exec_aten::Tensor ret_expected = tfFloat.make(
      {1, 1, 1, 8}, {-NAN, -NAN, -NAN, -NAN, -NAN, -NAN, -NAN, -NAN});
  exec_aten::Tensor ret = op_scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, out);
  EXPECT_TENSOR_CLOSE(ret, ret_expected);
}

TEST(OpScaledDotProductAttentionTest, CorrectnessTest_13) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 8, 1, 1}, {-47.0, 21.25, 74.75, 46.375, 21.0, -29.0, 2.625, 83.125});
  exec_aten::Tensor key = tfFloat.make(
      {1, 8, 3, 1},
      {-43.0,  12.5,    -68.125, -3.25,  -10.0,  65.0,    49.75,   -83.125,
       97.125, -40.375, -5.5,    93.125, 70.875, -67.375, -44.875, 98.25,
       -76.25, -74.5,   -23.25,  -66.75, 42.625, -88.0,   -37.75,  -61.625});
  exec_aten::Tensor value = tfFloat.make(
      {1, 8, 3, 1},
      {65.0,   81.125,  8.125,  68.375, -54.25, -1.125, -73.25, -54.0,
       -28.75, -23.875, 49.0,   63.5,   96.375, 16.625, 79.5,   33.125,
       32.875, -73.75,  69.125, 7.25,   -35.0,  94.0,   6.75,   65.75});
  exec_aten::optional<exec_aten::Tensor> attn_mask;
  double dropout_p = 0.0;
  bool is_causal = true;
  exec_aten::optional<double> scale;
  std::vector<int32_t> out_size(query.sizes().begin(), query.sizes().end());
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret_expected = tfFloat.make(
      {1, 8, 1, 1},
      {65.0, 68.375, -73.25, -23.875, 96.375, 33.125, 69.125, 94.0});
  exec_aten::Tensor ret = op_scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, out);
  EXPECT_TENSOR_CLOSE(ret, ret_expected);
}

TEST(OpScaledDotProductAttentionTest, CorrectnessTest_17) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {3, 2, 2, 6},
      {69.625,  -98.125, -22.0,   -17.25, -75.625, -43.875, -74.75,  14.5,
       82.0,    -82.625, 25.125,  -98.0,  -91.5,   65.875,  23.0,    50.25,
       30.125,  58.25,   -1.375,  23.0,   72.625,  47.875,  -76.125, -62.25,
       82.0,    -89.25,  75.25,   99.0,   -4.375,  -46.75,  94.875,  -16.375,
       -90.875, 81.875,  63.75,   -67.25, -13.625, 17.625,  -12.875, 86.0,
       10.875,  -57.625, 62.75,   -69.5,  -96.625, 80.0,    94.875,  17.5,
       -17.125, -69.5,   26.375,  25.5,   -51.625, 32.5,    15.0,    65.5,
       -49.0,   -71.25,  -18.625, -82.0,  94.25,   -56.25,  2.0,     21.25,
       37.125,  -9.0,    65.0,    -86.75, -77.0,   -26.75,  -99.875, -8.5});
  exec_aten::Tensor key = tfFloat.make(
      {3, 2, 4, 6},
      {98.125,  -86.25,  25.25,   -33.125, -98.0,   -42.5,   44.75,   42.375,
       -68.625, -97.375, 70.625,  0.75,    51.375,  89.75,   -62.5,   0.5,
       6.75,    92.875,  10.375,  -20.5,   20.75,   13.625,  -11.0,   99.0,
       52.75,   31.625,  -97.375, -51.0,   -31.25,  -78.5,   92.125,  -99.75,
       -10.5,   -39.125, 46.375,  98.5,    -81.5,   -61.375, 29.5,    -39.75,
       -54.875, 12.0,    80.25,   40.875,  58.25,   96.0,    -97.625, 31.625,
       63.625,  -3.875,  86.5,    -27.25,  8.875,   57.625,  88.375,  57.125,
       -17.5,   83.875,  84.75,   -27.375, 90.625,  -24.5,   76.5,    28.625,
       -71.625, 6.75,    -91.5,   -19.125, 24.5,    -76.0,   -6.5,    -77.625,
       46.625,  21.125,  -53.25,  -80.375, 59.0,    -21.125, -39.125, 90.75,
       -68.5,   -18.75,  44.625,  -44.75,  -24.0,   37.0,    -58.125, 13.25,
       -71.125, 16.875,  -4.625,  10.25,   12.375,  92.875,  76.0,    12.875,
       32.125,  94.5,    -58.25,  83.25,   -28.375, -27.875, 32.5,    -51.875,
       -94.75,  -65.5,   -48.875, 18.375,  -54.125, 52.625,  -51.0,   -66.125,
       64.5,    -31.0,   82.25,   42.0,    37.5,    -72.5,   66.625,  -96.5,
       59.375,  -69.625, -47.25,  -11.5,   -8.5,    -90.875, -64.75,  -61.75,
       97.0,    1.75,    -17.375, 99.875,  -85.375, 6.25,    41.625,  5.75,
       78.375,  -50.75,  9.75,    36.875,  84.5,    19.625,  -83.75,  17.0});
  exec_aten::Tensor value = tfFloat.make(
      {3, 2, 4, 6},
      {-26.375, -65.0,   55.5,    37.0,    90.0,    54.25,   83.75,   -33.75,
       2.375,   99.5,    71.5,    70.5,    -3.625,  -30.875, 46.125,  -60.5,
       -7.375,  -82.25,  42.5,    -3.125,  -9.25,   54.0,    -36.875, -67.875,
       -5.75,   -51.625, -8.875,  -36.25,  86.625,  84.5,    -28.75,  23.375,
       -39.625, 79.375,  95.0,    -51.125, -28.625, -82.375, 14.5,    -85.875,
       -92.125, 97.875,  -78.125, -34.0,   16.375,  -1.625,  70.375,  -58.625,
       96.75,   -95.125, -36.375, -72.875, 16.375,  -38.75,  -58.875, -97.0,
       -94.25,  -76.125, -30.0,   -60.0,   77.375,  34.75,   -16.5,   5.5,
       -16.25,  -40.75,  -7.625,  18.875,  -59.125, -56.0,   -7.25,   -14.375,
       -44.375, 87.625,  38.75,   79.5,    61.5,    29.375,  7.25,    -4.5,
       -46.25,  -88.875, -0.625,  -6.0,    -23.375, -18.25,  86.0,    33.375,
       60.25,   -23.125, 37.75,   5.5,     83.875,  -14.625, -89.75,  -84.625,
       -33.5,   90.5,    -53.125, 11.625,  90.875,  49.0,    -89.625, -6.75,
       -31.25,  -29.0,   -5.5,    72.5,    44.25,   66.0,    -76.75,  -7.375,
       52.375,  76.375,  -30.125, -72.875, 37.125,  -83.625, 60.875,  -98.125,
       -23.625, 85.875,  -25.875, 57.625,  50.75,   76.625,  -72.5,   26.0,
       65.875,  13.125,  -19.625, 7.5,     -25.5,   40.25,   75.25,   -48.0,
       8.25,    5.125,   42.375,  23.75,   65.25,   -77.0,   35.625,  -12.0});
  exec_aten::optional<exec_aten::Tensor> attn_mask;
  double dropout_p = 0.0;
  bool is_causal = false;
  exec_aten::optional<double> scale;
  exec_aten::Tensor ret_expected = tfFloat.make(
      {3, 2, 2, 6},
      {-26.375, -65.0,   55.5,    37.0,    90.0,    54.25,   83.75,   -33.75,
       2.375,   99.5,    71.5,    70.5,    -28.625, -82.375, 14.5,    -85.875,
       -92.125, 97.875,  -78.125, -34.0,   16.375,  -1.625,  70.375,  -58.625,
       77.375,  34.75,   -16.5,   5.5,     -16.25,  -40.75,  -58.875, -97.0,
       -94.25,  -76.125, -30.0,   -60.0,   37.75,   5.5,     83.875,  -14.625,
       -89.75,  -84.625, 37.75,   5.5,     83.875,  -14.625, -89.75,  -84.625,
       -89.625, -6.75,   -31.25,  -29.0,   -5.5,    72.5,    -30.125, -72.875,
       37.125,  -83.625, 60.875,  -98.125, -23.625, 85.875,  -25.875, 57.625,
       50.75,   76.625,  -23.625, 85.875,  -25.875, 57.625,  50.75,   76.625});
  std::vector<int32_t> out_size(query.sizes().begin(), query.sizes().end());
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, out);
  EXPECT_TENSOR_CLOSE(ret, ret_expected);
}

TEST(OpScaledDotProductAttentionTest, CorrectnessTest_18) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {3, 2, 2, 6},
      {44.0,    -13.875, -10.125, 36.625,  72.875,  -45.0,   87.5,    -5.375,
       25.25,   -28.625, 8.75,    -95.125, -75.5,   -59.25,  2.25,    -5.75,
       50.25,   83.375,  -19.0,   43.875,  -98.5,   43.375,  -27.875, 7.875,
       -15.875, 77.625,  92.5,    -16.375, -2.375,  20.25,   -75.875, -33.875,
       13.75,   9.875,   0.625,   78.5,    6.625,   -71.625, -38.25,  -33.5,
       -0.375,  -47.25,  55.875,  -49.0,   66.25,   88.625,  -28.75,  -49.75,
       -6.5,    23.5,    -84.875, -13.25,  4.875,   -2.125,  -56.25,  85.75,
       44.5,    -78.75,  -39.875, 31.0,    -73.125, 68.875,  -42.625, 29.75,
       35.125,  83.0,    29.625,  89.75,   64.875,  91.875,  40.375,  -92.75});
  exec_aten::Tensor key = tfFloat.make(
      {3, 2, 4, 6},
      {-11.375, -70.5,   10.125,  -76.125, -26.5,   -11.375, -1.125,  7.5,
       94.375,  -50.125, 43.125,  61.75,   39.375,  -79.25,  41.375,  88.75,
       -72.625, -17.125, 48.0,    80.75,   -66.125, -8.625,  -41.0,   6.75,
       -37.75,  91.375,  4.0,     27.625,  51.625,  80.5,    -64.5,   21.875,
       89.0,    -71.625, 32.75,   29.25,   -70.625, 6.875,   -1.75,   55.875,
       -19.125, -99.125, -73.0,   -62.75,  -17.25,  37.625,  -86.75,  58.75,
       -40.75,  45.125,  -38.5,   -60.125, 90.625,  99.875,  71.25,   -88.625,
       74.625,  42.0,    -75.875, 57.375,  -29.0,   -25.75,  72.5,    76.875,
       -27.0,   -2.625,  -26.375, -94.0,   -71.625, -18.125, -25.875, -62.0,
       7.625,   73.125,  -87.625, 98.875,  -61.25,  -96.75,  -25.625, -57.875,
       53.75,   68.25,   84.125,  36.125,  38.125,  -82.375, 92.5,    -82.75,
       -91.25,  -60.25,  -46.375, 79.625,  20.25,   13.125,  -54.125, -32.625,
       -35.25,  -51.75,  13.625,  -62.375, 91.0,    -45.5,   85.125,  17.625,
       99.5,    8.875,   -92.75,  81.375,  18.625,  37.625,  0.75,    23.125,
       -81.5,   76.75,   10.875,  40.125,  -22.875, -24.875, 52.5,    0.875,
       59.25,   48.125,  40.875,  -43.25,  -65.625, -27.25,  58.0,    91.125,
       78.625,  45.875,  76.0,    79.375,  17.0,    9.75,    -26.75,  15.0,
       -1.0,    -84.75,  -38.5,   -50.625, -68.375, 0.375,   -47.0,   -91.75});
  exec_aten::Tensor value = tfFloat.make(
      {3, 2, 4, 6},
      {-39.25,  69.875, -28.125, 18.375,  89.375,  -39.5,   -55.25,  -42.0,
       -7.875,  26.625, -6.125,  -98.25,  48.625,  33.625,  48.0,    96.75,
       -59.125, -85.25, -22.25,  -91.0,   -1.75,   14.25,   -7.75,   -94.375,
       -97.625, 71.0,   90.875,  -11.5,   -14.625, 52.875,  90.875,  32.875,
       -84.25,  -57.75, -78.875, -81.75,  86.0,    54.125,  -75.625, -28.375,
       24.375,  45.125, 80.375,  -42.25,  3.5,     -68.5,   2.875,   58.75,
       9.625,   -52.75, -31.25,  74.25,   -98.0,   38.0,    59.25,   45.5,
       67.75,   52.5,   -59.75,  20.0,    83.875,  -46.75,  5.25,    74.375,
       14.125,  -67.0,  -60.625, 28.5,    20.5,    -96.625, -89.125, 33.875,
       -89.25,  9.875,  -99.25,  -20.5,   78.625,  37.875,  -72.375, -49.625,
       22.0,    -54.25, 18.125,  57.75,   72.375,  -11.5,   -52.5,   -28.125,
       -86.875, -45.0,  60.25,   34.625,  -88.875, 91.0,    -48.25,  98.75,
       100.0,   33.0,   -69.625, -88.25,  -46.625, -24.75,  -77.5,   93.5,
       -45.125, 42.75,  -50.0,   -86.0,   -17.375, 85.25,   -28.125, -28.375,
       46.375,  26.625, 23.0,    -55.875, 39.125,  87.25,   -9.625,  95.375,
       -27.875, 59.5,   15.5,    -90.0,   39.5,    -15.75,  -16.375, -96.875,
       -96.125, -47.0,  0.75,    -45.875, 74.625,  46.0,    20.5,    -42.875,
       -55.0,   30.375, -27.375, 99.375,  18.375,  0.375,   54.25,   -57.75});
  exec_aten::optional<exec_aten::Tensor> attn_mask;
  double dropout_p = 0.0;
  bool is_causal = false;
  exec_aten::optional<double> scale = exec_aten::optional<double>(-INFINITY);
  exec_aten::Tensor ret_expected = tfFloat.make(
      {3, 2, 2, 6},
      {NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
       NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
       NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
       NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
       NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN,
       NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
  std::vector<int32_t> out_size(query.sizes().begin(), query.sizes().end());
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, out);
  EXPECT_TENSOR_CLOSE(ret, ret_expected);
}

/*
// Disabling this test because right now we are enforcing that
// attention mask must be 2D
TEST(OpScaledDotProductAttentionTest, CorrectnessTest_19) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {3, 2, 2, 6},
      {-50.875, 17.375,  -42.875, 8.125,   -59.625, -59.125, 0.0,     -76.375,
       39.625,  -27.75,  -43.375, 71.0,    -96.5,   -48.75,  23.125,  11.125,
       30.125,  36.75,   -22.25,  35.625,  37.875,  -43.375, -22.875, 74.75,
       79.375,  -75.25,  66.75,   48.875,  88.875,  -73.5,   79.375,  55.5,
       -84.0,   93.0,    -19.625, -49.875, 88.625,  -5.0,    -94.625, -13.375,
       88.375,  -30.625, 39.75,   -15.625, -80.5,   -40.25,  -90.375, -0.5,
       -47.625, 86.875,  -27.125, 26.75,   41.0,    48.0,    4.375,   10.125,
       -26.375, 4.25,    56.5,    -45.625, -78.75,  99.625,  -5.5,    -85.0,
       18.125,  -71.5,   6.0,     -44.125, 59.125,  49.25,   21.125,  -6.5});
  exec_aten::Tensor key = tfFloat.make(
      {3, 2, 4, 6},
      {-36.25,  -6.125,  49.0,    -14.375, 22.25,   17.75,   69.125,  22.625,
       -0.125,  -85.875, -71.125, -1.375,  -43.75,  -55.25,  71.125,  58.375,
       19.875,  -98.0,   -16.875, -29.375, 83.875,  19.125,  -18.5,   -34.75,
       -59.75,  -92.625, -19.375, 55.625,  -1.75,   25.0,    82.25,   8.0,
       6.75,    28.5,    8.125,   -24.375, 52.875,  -39.75,  66.625,  -31.375,
       -42.25,  -30.25,  -20.875, 24.75,   -34.5,   -69.75,  -9.0,    65.625,
       42.125,  89.5,    -1.875,  -88.375, 82.375,  80.25,   7.875,   71.0,
       84.125,  -9.625,  -62.0,   7.625,   83.0,    55.0,    -65.125, -55.125,
       -10.0,   17.75,   67.0,    83.25,   51.125,  -13.75,  40.875,  -77.625,
       19.125,  -48.125, -86.125, -20.5,   -93.125, 64.5,    -5.5,    72.375,
       86.625,  -21.0,   77.0,    -85.625, 14.5,    69.75,   99.875,  -14.125,
       36.875,  -50.375, -65.5,   94.5,    64.0,    61.0,    -73.0,   -24.375,
       -11.5,   -16.75,  92.0,    62.5,    62.375,  -81.625, -25.125, -53.25,
       -61.375, 58.5,    -67.625, 26.5,    64.0,    27.25,   84.5,    4.125,
       -82.375, 2.0,     21.5,    0.75,    80.0,    -87.375, 38.75,   -25.25,
       68.75,   -18.875, 74.75,   -45.625, -15.875, 13.5,    51.25,   37.25,
       -12.0,   -15.5,   -45.75,  7.375,   1.25,    -54.375, 80.25,   18.875,
       89.0,    -30.625, -39.5,   -39.0,   46.625,  -46.0,   -87.125, -18.0});
  exec_aten::Tensor value = tfFloat.make(
      {3, 2, 4, 6},
      {-74.5,   -0.25,   -77.125, -74.375, -53.0,   33.625,  -45.0,   66.0,
       -66.875, -71.875, -9.75,   -41.125, 37.0,    -65.25,  -50.25,  84.75,
       -67.875, 54.0,    16.875,  -96.5,   91.75,   14.625,  80.875,  -25.875,
       -62.75,  -92.5,   -77.75,  40.75,   -53.125, -71.875, 10.0,    -4.75,
       -54.875, -24.25,  48.625,  9.375,   -9.625,  32.875,  -62.75,  99.5,
       25.125,  85.625,  -29.0,   -33.75,  44.0,    -83.75,  44.125,  -88.625,
       -17.75,  22.625,  -79.5,   1.0,     -10.625, 10.0,    70.25,   -91.625,
       -86.0,   83.875,  68.25,   -35.125, -6.25,   -81.25,  -38.375, 56.0,
       26.875,  -51.75,  -79.625, 83.375,  -31.625, 83.375,  -4.75,   81.875,
       53.0,    -31.625, -48.625, 76.75,   71.625,  -63.0,   17.25,   -22.0,
       -7.75,   -77.25,  -92.25,  -2.0,    -88.0,   88.5,    -54.125, -7.875,
       98.0,    -56.75,  96.125,  -90.0,   -70.0,   -50.125, -53.5,   -65.125,
       48.375,  98.125,  -89.0,   -97.125, 20.625,  85.5,    -77.625, 76.0,
       73.625,  58.625,  -90.375, 11.75,   -16.5,   78.125,  95.375,  86.375,
       -69.125, -92.375, -65.25,  27.875,  77.125,  -59.875, 79.5,    -78.625,
       15.25,   53.75,   44.625,  -22.0,   -84.0,   -7.25,   22.0,    25.875,
       17.625,  -86.875, 22.75,   -74.0,   -79.875, -68.0,   -71.125, -81.625,
       -4.125,  65.875,  1.875,   76.125,  -43.75,  -15.25,  -4.625,  -66.125});
  exec_aten::optional<exec_aten::Tensor> attn_mask =
      exec_aten::optional<exec_aten::Tensor>(tfFloat.make(
          {3, 1, 2, 2, 4},
          {39.0,  49.375,  -87.125, -99.125, 49.375,  -41.125, 26.25,   79.75,
           91.0,  -3.125,  65.75,   63.5,    -48.375, 43.375,  22.5,    -53.625,
           -70.0, 2.125,   21.875,  6.375,   -6.375,  75.25,   -35.875, 86.375,
           71.5,  -35.875, 19.75,   11.625,  -87.25,  49.0,    -6.0,    62.875,
           7.125, 87.375,  -14.75,  55.5,    59.125,  24.75,   -66.5,   72.375,
           2.25,  81.375,  -87.125, 35.125,  -39.125, 43.5,    52.875,  39.5}));
  double dropout_p = 0.0;
  bool is_causal = false;
  exec_aten::optional<double> scale;
  exec_aten::Tensor ret_expected = tfFloat.make(
      {3, 1, 2, 2, 6},
      {37.0,
       -65.25,
       -50.25,
       84.75,
       -67.875,
       54.0,
       16.874713897705078,
       -96.4992446899414,
       91.749267578125,
       14.624600410461426,
       80.87458038330078,
       -25.87506866455078,
       -62.75,
       -92.5,
       -77.75,
       40.75,
       -53.125,
       -71.875,
       -29.0,
       -33.75,
       44.0,
       -83.75,
       44.125,
       -88.625,
       -79.625,
       83.375,
       -31.625,
       83.375,
       -4.75,
       81.875,
       -6.25,
       -81.25,
       -38.375,
       56.0,
       26.875,
       -51.75,
       17.25,
       -22.0,
       -7.75,
       -77.25,
       -92.25,
       -2.0,
       53.0,
       -31.625,
       -48.625,
       76.75,
       71.625,
       -63.0,
       -77.625,
       76.0,
       73.625,
       58.625,
       -90.375,
       11.75,
       48.375,
       98.125,
       -89.0,
       -97.125,
       20.625,
       85.5,
       1.875,
       76.125,
       -43.75,
       -15.25,
       -4.625,
       -66.125,
       -79.875,
       -68.0,
       -71.125,
       -81.625,
       -4.125,
       65.875});
  Tensor ret = op_scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale);
  EXPECT_TENSOR_CLOSE(ret, ret_expected);
}
*/

TEST(OpScaledDotProductAttentionTest, CorrectnessTest_51) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 1, 8, 3},
      {-14.0,  46.125, -78.125, -61.375, 52.375, -9.125, 57.875, 88.25,
       -95.75, 8.875,  -64.625, 41.75,   -62.25, 41.25,  -67.25, 51.25,
       48.0,   67.625, 30.0,    -59.0,   42.25,  -33.0,  -10.25, -77.5});
  exec_aten::Tensor key = tfFloat.make(
      {1, 1, 3, 3},
      {6.0, 58.5, -37.875, -11.125, -18.5, 35.0, 59.25, 73.0, 34.125});
  exec_aten::Tensor value = tfFloat.make(
      {1, 1, 3, 3},
      {70.375, 30.875, 72.125, 53.0, 39.125, -4.625, 26.5, 79.5, 88.625});
  exec_aten::optional<exec_aten::Tensor> attn_mask =
      exec_aten::optional<exec_aten::Tensor>(tfFloat.make(
          {8, 3},
          {-59.25, -26.25, -3.0,  -24.125, 47.75,  92.375,  87.5,    21.5,
           64.5,   45.0,   -54.0, 17.375,  -67.75, 14.625,  88.75,   36.0,
           88.375, 25.75,  42.5,  -13.375, -82.75, -59.625, -21.125, 6.5}));
  double dropout_p = 0.0;
  bool is_causal = false;
  exec_aten::optional<double> scale;
  exec_aten::Tensor ret_expected = tfFloat.make(
      {1, 1, 8, 3},
      {70.375, 30.875, 72.125, 70.375, 30.875, 72.125, 70.375, 30.875,
       72.125, 53.0,   39.125, -4.625, 70.375, 30.875, 72.125, 26.5,
       79.5,   88.625, 53.0,   39.125, -4.625, 70.375, 30.875, 72.125});
  std::vector<int32_t> out_size(query.sizes().begin(), query.sizes().end());
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, out);
  EXPECT_TENSOR_CLOSE(ret, ret_expected);
}
