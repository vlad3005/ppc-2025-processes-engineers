#include "borunov_v_cnt_words/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <cctype>   // Для std::isspace
#include <sstream>
#include <string>
#include <vector>

namespace borunov_v_cnt_words {
BorunovVCntWordsMPI::BorunovVCntWordsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BorunovVCntWordsMPI::ValidationImpl() {
  return GetOutput() == 0;
}

bool BorunovVCntWordsMPI::PreProcessingImpl() {
  return true;
}

bool BorunovVCntWordsMPI::RunImpl() {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string full_text;
    if (rank == 0) {
        full_text = GetInput();
    }


    int text_len = (rank == 0) ? static_cast<int>(full_text.size()) : 0;
    MPI_Bcast(&text_len, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (rank != 0) full_text.resize(text_len);

 
    MPI_Bcast(full_text.data(), text_len, MPI_CHAR, 0, MPI_COMM_WORLD);


    std::vector<int> send_counts(world_size, 0);
    std::vector<int> displs(world_size, 0);

    if (rank == 0) {
        if (world_size > 0 && text_len > 0) {
            int base = text_len / world_size;
            int extra = text_len % world_size;
            int off = 0;
            for (int i = 0; i < world_size; ++i) {
                send_counts[i] = base + (i < extra ? 1 : 0);
                displs[i] = off;
                off += send_counts[i];
            }
        } else {
            // текст пустой или один процесс — все нули
            for (int i = 0; i < world_size; ++i) {
                send_counts[i] = 0;
                displs[i] = 0;
            }
        }
    }


    MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);


    int start = send_counts.empty() ? 0 : displs[rank];
    int len = send_counts.empty() ? 0 : send_counts[rank];
    int end = start + len; // exclusive


    unsigned long long local_count = 0;
    for (int i = start; i < end; ++i) {
        unsigned char curr = static_cast<unsigned char>(full_text[i]);
        if (!std::isspace(curr)) {

            bool is_word_start = false;
            if (i == start) {
                if (start == 0) {
                    is_word_start = true;
                } else {
                    unsigned char left = static_cast<unsigned char>(full_text[start - 1]);
                    if (std::isspace(left)) is_word_start = true;
                }
            } else {
                unsigned char left = static_cast<unsigned char>(full_text[i - 1]);
                if (std::isspace(left)) is_word_start = true;
            }

            if (is_word_start) local_count++;
        }
    }

    unsigned long long global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Bcast(&global_count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    GetOutput() = static_cast<OutType>(global_count);

    MPI_Barrier(MPI_COMM_WORLD);
    return true;
}

bool BorunovVCntWordsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_cnt_words