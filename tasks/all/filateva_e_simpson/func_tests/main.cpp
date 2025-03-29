#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "all/filateva_e_simpson/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

TEST(filateva_e_simpson_all, test_x_pow_2) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {1};
    b = {10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return x[0] * x[0] * x[0] / 3;
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x_pow_2_negative) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {-10};
    b = {10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return x[0] * x[0] * x[0] / 3;
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {1};
    b = {10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return x[0] * x[0] / 2;
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x_pow_3) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {1};
    b = {100};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return std::pow(x[0], 4) / 4;
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x_del) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return 1 / x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {1};
    b = {10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return std::log(x[0]);
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x_sin) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return std::sin(x[0]);
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {1};
    b = {std::numbers::pi};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return -std::cos(x[0]);
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x_cos) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return std::cos(x[0]);
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {1};
    b = {std::numbers::pi / 2};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return std::sin(x[0]);
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_gausa) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return pow(std::numbers::e, -pow(x[0], 2));
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {0};
    b = {1};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], 0.746824, 0.01);
  }
}

TEST(filateva_e_simpson_all, test_sum_integral) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return pow(x[0], 3) + pow(x[0], 2) + x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {0};
    b = {10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    filateva_e_simpson_all::Func integral_f = [](std::vector<double> x) {
      if (x.empty()) {
        return 0.0;
      }
      return (pow(x[0], 4) / 4) + (pow(x[0], 3) / 3) + (pow(x[0], 2) / 2);
    };

    ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
  }
}

TEST(filateva_e_simpson_all, test_error_1) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 100;
    a = {10};
    b = {0};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  if (world.rank() == 0) {
    ASSERT_FALSE(simpson.Validation());
  } else {
    ASSERT_TRUE(simpson.Validation());
  }
}

TEST(filateva_e_simpson_all, test_error_n_mer) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 3;
    steps = 100;
    a = {0, 0, 10};
    b = {10, 10, 0};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  if (world.rank() == 0) {
    ASSERT_FALSE(simpson.Validation());
  } else {
    ASSERT_TRUE(simpson.Validation());
  }
}

TEST(filateva_e_simpson_all, test_error_2) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 1;
    steps = 101;
    a = {0};
    b = {10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  if (world.rank() == 0) {
    ASSERT_FALSE(simpson.Validation());
  } else {
    ASSERT_TRUE(simpson.Validation());
  }
}

TEST(filateva_e_simpson_all, test_x_y_pow_2) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return (param[0] * param[0]) + (param[1] * param[1]);
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 2;
    steps = 100;
    a = {0, 0};
    b = {1, 1};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], 0.66666, 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x_y) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return param[0] + param[1];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 2;
    steps = 100;
    a = {0, 0};
    b = {10, 10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], 1000, 0.01);
  }
}

TEST(filateva_e_simpson_all, test_sin_x_cos_y) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return std::sin(param[0]) * std::cos(param[1]);
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 2;
    steps = 100;
    a = {0, 0};
    b = {std::numbers::pi, std::numbers::pi / 2};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], 2, 0.01);
  }
}

TEST(filateva_e_simpson_all, test_sum_integral_x_y) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return pow(param[0], 3) + pow(param[1], 2) + param[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 2;
    steps = 100;
    a = {0, 0};
    b = {10, 10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], 28833.33, 0.01);
  }
}

TEST(filateva_e_simpson_all, test_x_y_negative) {
  boost::mpi::communicator world;
  size_t mer;
  size_t steps;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> res;
  filateva_e_simpson_all::Func f = [](std::vector<double> param) {
    if (param.empty()) {
      return 0.0;
    }
    return param[0] + param[1];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    mer = 2;
    steps = 100;
    a = {-10, -10};
    b = {10, 10};
    res.resize(1, 0);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    task_data->inputs_count.emplace_back(mer);
    task_data->inputs_count.emplace_back(steps);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data->outputs_count.emplace_back(1);
  }

  filateva_e_simpson_all::Simpson simpson(task_data);
  simpson.setFunc(f);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  if (world.rank() == 0) {
    ASSERT_NEAR(res[0], 0, 0.01);
  }
}
