#ifndef STATIC_ASSERT_H
#define STATIC_ASSERT_H

#include <Eigen/Eigen>

// static assertion failing if the type \a TYPE is not a vector type of the given size
#define EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE_OR_DYNAMIC(TYPE, SIZE) \
  EIGEN_STATIC_ASSERT(TYPE::IsVectorAtCompileTime && (TYPE::SizeAtCompileTime == SIZE || TYPE::SizeAtCompileTime == Eigen::Dynamic), THIS_METHOD_IS_ONLY_FOR_VECTORS_OF_A_SPECIFIC_SIZE)

// static assertion failing if the type \a TYPE is not a vector type of the given size
#define EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE_OR_DYNAMIC(TYPE, ROWS, COLS)                                                                                                           \
  EIGEN_STATIC_ASSERT((TYPE::RowsAtCompileTime == ROWS || TYPE::RowsAtCompileTime == Eigen::Dynamic) && (TYPE::ColsAtCompileTime == COLS || TYPE::RowsAtCompileTime == Eigen::Dynamic), \
                      THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE)

// static assert
#define EIGEN_STATIC_ASSERT_SAME_TYPE(U, V, MSG) EIGEN_STATIC_ASSERT((sm::common::SameType<U, V>::value), MSG)

#endif // STATIC_ASSERT_H