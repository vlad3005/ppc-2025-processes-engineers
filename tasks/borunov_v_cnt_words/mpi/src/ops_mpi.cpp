#include "borunov_v_cnt_words/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>  // Для std::isspace
#include <cstdint>
#include <string>
#include <vector>

#include "borunov_v_cnt_words/common/include/common.hpp"

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

void BorunovVCntWordsMPI::CalculateChunks(int text_len, int world_size, std::vector<int> &send_counts,
                                          std::vector<int> &displs) {
  send_counts.resize(world_size, 0);
  displs.resize(world_size, 0);

  // Устраняет сложный if/else-if/for блок, снижая сложность RunImpl.
  if (text_len <= 0 || world_size <= 0) {
    // Если текст пуст или 0 процессов, массивы остаются заполненными нулями (по resize)
    return;
  }

  int base = text_len / world_size;
  int extra = text_len % world_size;
  int off = 0;
  for (int i = 0; i < world_size; ++i) {
    // Здесь используется условный оператор, но он локализован
    send_counts[i] = base + (i < extra ? 1 : 0);
    displs[i] = off;
    off += send_counts[i];
  }
}

// 2. Вынесение сложной логики подсчета слов на границах
uint64_t BorunovVCntWordsMPI::ComputeLocalCount(const std::string &full_text, int start, int end) {
  uint64_t local_count = 0;

  // Внутренняя логика подсчета слов
  for (int i = start; i < end; ++i) {
    // Исправлено: modernize-use-auto
    auto curr = static_cast<unsigned char>(full_text[i]);

    // Исправлено: implicit conversion 'int' -> 'bool'
    if (!static_cast<bool>(std::isspace(curr))) {
      bool is_word_start;

      if (i == start) {
        if (start == 0) {
          is_word_start = true;
        } else {
          // Исправлено: modernize-use-auto
          auto left = static_cast<unsigned char>(full_text[start - 1]);
          // Исправлено: implicit conversion 'int' -> 'bool'
          is_word_start = static_cast<bool>(std::isspace(left));
        }
      } else {
        // Исправлено: modernize-use-auto
        auto left = static_cast<unsigned char>(full_text[i - 1]);
        // Исправлено: implicit conversion 'int' -> 'bool'
        is_word_start = static_cast<bool>(std::isspace(left));
      }

      if (is_word_start) {
        local_count++;
      }
    }
  }
  return local_count;
}

bool BorunovVCntWordsMPI::RunImpl() {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::string full_text;

  // A. Чтение и Bcast данных (Логика rank == 0 вынесена, но Bcast остается)
  if (rank == 0) {
    full_text = GetInput();
  }

  // Исправлено: Условный оператор заменен на if/else-if для снижения сложности
  // Примечание: Условный оператор (тернарный) считается +1, оставляем его,
  // но заменяем его на чистый код без вложенности.
  int text_len = 0;
  if (rank == 0) {
    text_len = static_cast<int>(full_text.size());
  }

  MPI_Bcast(&text_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Ранний выход для инициализации, избегая if (rank != 0) { ... }
  if (rank != 0 && text_len > 0) {
    full_text.resize(text_len);
  }

  MPI_Bcast(full_text.data(), text_len, MPI_CHAR, 0, MPI_COMM_WORLD);

  // B. Расчет и Bcast размеров чанков (Логика расчета вынесена)
  std::vector<int> send_counts;
  std::vector<int> displs;

  if (rank == 0) {
    CalculateChunks(text_len, world_size, send_counts, displs);
  }

  // Убедимся, что все процессы имеют векторы нужного размера перед Bcast
  send_counts.resize(world_size, 0);
  displs.resize(world_size, 0);

  MPI_Bcast(send_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  // C. Вычисление локального счетчика (Логика подсчета вынесена)
  // Условный оператор заменен на явный if для снижения сложности (если это требуется)
  int start = 0;
  int len = 0;
  if (world_size > 0 && rank < world_size) {
    start = displs[rank];
    len = send_counts[rank];
  }
  int end = start + len;

  // Исправлено: google-runtime-int
  uint64_t local_count = ComputeLocalCount(full_text, start, end);

  // D. Reduce и Bcast итогов
  // Исправлено: google-runtime-int
  uint64_t global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  // Исправлено: no header providing "borunov_v_cnt_words::OutType"
  // (Требуется включение заголовка 'common.hpp' или аналогичного)
  GetOutput() = static_cast<OutType>(global_count);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BorunovVCntWordsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace borunov_v_cnt_words
