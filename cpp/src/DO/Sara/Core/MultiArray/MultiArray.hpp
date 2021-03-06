// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! @brief This contains the implementation of the N-dimensional array class.

#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>


namespace DO { namespace Sara {

  //! @brief The multidimensional array class.
  template <typename MultiArrayView, template <typename> class Allocator>
  class MultiArrayBase : public MultiArrayView
  {
    //! @{
    //! Convenience typedefs.
    using self_type =  MultiArrayBase;
    using base_type = MultiArrayView;
    //! @}

    using base_type::_begin;
    using base_type::_end;
    using base_type::_sizes;
    using base_type::_strides;

  public:
    using base_type::Dimension;
    using base_type::StorageOrder;

    using value_type = typename base_type::value_type;
    using pointer = typename base_type::pointer;
    using vector_type = typename base_type::vector_type;
    using allocator_type = Allocator<value_type>;

  public: /* interface */
    //! @brief Default constructor that constructs an empty ND-array.
    inline MultiArrayBase() = default;

    //! @{
    //! @brief Constructor with specified sizes.
    inline explicit MultiArrayBase(const vector_type& sizes)
    {
      initialize(sizes);
    }

    inline explicit MultiArrayBase(int rows, int cols)
      : self_type{vector_type{rows, cols}}
    {
    }

    inline explicit MultiArrayBase(int rows, int cols, int depth)
      : self_type{vector_type{rows, cols, depth}}
    {
    }
    //! @}

    //! @brief Copy constructor.
    //! Clone the other MultiArrayView instance.
    inline MultiArrayBase(const base_type& other)
    {
      initialize(other.sizes());
      base_type::copy(other);
    }

    inline MultiArrayBase(const self_type& other)
      : self_type{base_type(other)}
    {
    }

    //! @brief Move constructor.
    inline MultiArrayBase(self_type&& other)
    {
      base_type::swap(other);
    }

    //! @brief Destructor.
    inline ~MultiArrayBase()
    {
      deallocate();
    }

    //! @brief Assignment operator uses the copy-swap idiom.
    self_type& operator=(self_type other)
    {
      base_type::swap(other);
      return *this;
    }

    //! @{
    //! @brief Resize the MultiArray object with the specified sizes.
    inline void resize(const vector_type& sizes)
    {
      // Boundary case: the size of the allocated memory has not changed.
      // Optimize by:
      // - not deallocating memory.
      // - only changing the sizes and recompute the strides.
      if (_end - _begin == std::ptrdiff_t(base_type::compute_size(sizes)))
      {
        _sizes = sizes;
        _strides = base_type::compute_strides(sizes);
        return;
      }

      // General case.
      if (_sizes != sizes)
      {
        deallocate();
        initialize(sizes);
      }
    }

    inline void resize(int rows, int cols)
    {
      static_assert(Dimension == 2, "MultiArray must be 2D");
      resize(vector_type{rows, cols});
    }

    inline void resize(int rows, int cols, int depth)
    {
      static_assert(Dimension == 3, "MultiArray must be 3D");
      resize(vector_type(rows, cols, depth));
    }
    //! @}

    //! @brief Destroy the content of the MultiArray object.
    inline void clear()
    {
      deallocate();
    }

  private: /* helper functions for offset computation. */
    //! @{
    //! @brief Allocate the internal array of the MultiArray object.
    inline void initialize(const vector_type& sizes)
    {
      const auto empty = (sizes == vector_type::Zero());
      const auto num_elements = empty ? 0 : base_type::compute_size(sizes);

      _sizes = sizes;
      _strides = empty ? sizes : base_type::compute_strides(sizes);
      _begin = empty ? 0 : allocate(num_elements);
      _end = empty ? 0 : _begin + num_elements;
    }

    inline pointer allocate(std::size_t count)
    {
      return allocator_type{}.allocate(count);
    }
    //! @}

    //! @brief Deallocate the MultiArray object.
    inline void deallocate()
    {
      allocator_type{}.deallocate(_begin, _end - _begin);
      _begin = nullptr;
      _end = nullptr;
      _sizes = vector_type::Zero();
      _strides = vector_type::Zero();
    }

  };


  //! @}

} /* namespace Sara */
} /* namespace DO */
