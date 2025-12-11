#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "partition.h"

static void read_means(const std::string& means_path, uint32_t& out_nvec,
                       uint32_t& out_dim, std::vector<uint8_t>& out_buf) {
  std::ifstream means_file{means_path};

  means_file.read(reinterpret_cast<char*>(&out_nvec), sizeof(uint32_t));
  means_file.read(reinterpret_cast<char*>(&out_dim), sizeof(uint32_t));

  size_t buf_size = static_cast<size_t>(out_nvec) * out_dim * sizeof(float);

  out_buf.resize(buf_size);
  means_file.read(reinterpret_cast<char*>(out_buf.data()), buf_size);
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "usage: " << argv[0]
              << " <data_file> <means_file> <k_base> <out_dir>\n";
    return 1;
  }

  std::string data_path = argv[1];
  std::string means_path = argv[2];
  size_t k_base = std::strtoull(argv[3], NULL, 10);
  std::string out_dir = argv[4];

  if (k_base == 0) {
    std::cout << "invalid k_base\n";
    return 1;
  }

  // TODO: Adapt separators on other OSs?
  if (out_dir.back() != '/') {
    out_dir.push_back('/');
  }

  size_t last_sep = data_path.rfind("/");

  std::string data_filename;
  if (last_sep == std::string::npos) {
    data_filename = data_path;
  } else {
    data_filename = data_path.substr(last_sep + 1);
  }

  if (data_filename.length() == 0) {
    std::cout << "invalid data file path\n";
    return 1;
  }

  uint32_t mean_count;
  uint32_t mean_dim;
  std::vector<uint8_t> means_buf;

  read_means(means_path, mean_count, mean_dim, means_buf);

  // DO NOT move above `read_means` as it resizes the vector which may
  // invalidate the pointer.
  float* means = reinterpret_cast<float*>(means_buf.data());

  std::filesystem::create_directory(out_dir);

  if (shard_data_into_clusters<float>(data_path, means, mean_count, mean_dim,
                                      k_base, out_dir + data_filename) == -1) {
    std::cout << "cannot shard data into clusters\n";
    return -1;
  }

  return 0;
}
