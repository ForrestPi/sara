// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <tuple>

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/ImageIO/Database/ImageDataSet.hpp>
#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>


namespace DO { namespace Sara {

  template <typename XYTHandleIterator, typename XIterator, typename YIterator,
            typename DataTransformIterator>
  class TransformedTrainingDataSetIterator
  {
  public:
    using xyt_handle_iterator = XYTHandleIterator;
    using x_iterator = XPointer;
    using y_iterator = YPointer;
    using data_transform_pointer = DataTransformPointer;

    using x_type = typename XPointer::value_type;
    using y_type = typename YPointer::value_type;
    using data_transform_type = typename DataTransformPointer::value_type;
    using self_type = TransformedTrainingDataSetIterator;

    inline TransformedTrainingDataSetIterator() = default;

    inline TransformedTrainingDataSetIterator(xyt_handle_iterator xyt_handle_i,
                                              x_pointer x, y_pointer y,
                                              data_transform_pointer t)
      : _xyt_handle_i{xyt_handle_i}
      , _x{x}
      , _y{y}
      , _t{t}
    {
    }

    inline auto operator++() -> self_type&
    {
      ++_xyt_handle_i;
      return *this;
    }

    inline auto operator--() -> self_type&
    {
      --_xyt_handle_i;
      return *this;
    }

    inline auto operator+=(std::ptrdiff_t n) -> self_type&
    {
      _xyt_handle_i += n;
      return *this;
    }

    inline auto operator-=(std::ptrdiff_t n) -> self_type&
    {
      _xyt_handle_i -= n;
      return *this;
    }

    inline auto x_ptr() const -> x_pointer
    {
      return _x + std::get<0>(*_xyt_handle_i);
    }

    inline auto y_ptr() const -> y_pointer
    {
      return _y + std::get<0>(*_xyt_handle_i);
    }

    inline auto t_ptr() const -> data_transform_pointer
    {
      return _t + std::get<1>(*_xyt_handle_i);
    }

    inline auto x() const -> const x_type&
    {
      return *x_ptr();
    }

    inline auto y() const -> const y_type&
    {
      return *y_ptr();
    }

    inline auto t() const -> const data_transform_type&
    {
      return *t_ptr();
    }

    inline auto operator==(const self_type& other) const -> bool
    {
      return std::make_tuple(_xyt_handle_i, _x, _y, _t) ==
             std::make_tuple(other._xyt_handle_i, other._x, other._y, other._t);
    }

    inline auto operator!=(const self_type& other) const -> bool
    {
      return !(operator==(other));
    }

  private:
    xyt_handle_iterator _xyt_handle_i;
    x_pointer _x;
    y_pointer _y;
    data_transform_pointer _t;
  };


  template <typename XHandle, typename YHandle, typename DataTransform>
  class TransformedTrainingDataSet
  {
  public:
    using x_handle = XHandle;
    using y_handle = YHandle;
    using data_transform_type = DataTransform;

    using x_handle_array = std::vector<XHandle>;
    using y_handle_array = std::vector<YHandle>;
    using data_transform_array = std::vector<DataTransform>;

    using xyt_handle_array = std::vector<std::array<std::size_t, 2>>;

    inline TransformedTrainingDataSet() = default;

    inline void clear()
    {
      xyt.clear();
      x.clear();
      y.clear();
      t.clear();
    }

    inline bool operator==(const TransformedTrainingDataSet& other) const
    {
      return x == other.x && y == other.y && t == other.t && xyt = other.xyt;
    }

    inline bool operator!=(const TransformedTrainingDataSet& other) const
    {
      return !(*this == other);
    }

    xyt_handle_array xyt;
    x_handle_array x;
    y_handle_array y;
    data_transform_array t;
  };


  class TransformedImageClassificationTrainingDataSet
      : public TransformedTrainingDataSet<std::string, int, ImageDataTransform>
  {
    using base_type =
        TransformedTrainingDataSet<std::string, int, ImageDataTransform>;

    using base_type::xyt;
    using base_type::x;
    using base_type::y;
    using base_type::t;

  public:
    using x_iterator = ImageDataSetIterator<Image<Rgb8>>;
    using y_iterator = typename y_handle_array::const_iterator;
    using data_transform_iterator =
        typename data_transform_array::const_iterator;

    using iterator =
        TransformedTrainingDataSetIterator<xyt_handle_array::const_iterator,
                                           x_iterator, y_iterator,
                                           data_transform_iterator>;


    inline TransformedImageClassificationTrainingDataSet() = default;

    inline auto begin() const -> iterator
    {
      return iterator{xyt.begin(), x.begin(), y.begin(), t.begin()};
    }

    inline auto end() const -> iterator
    {
      return iterator{xyt.end(), x.end(), y.end(), t.end()};
    }

    auto x_begin() const -> x_iterator
    {
      return x_iterator{x.begin(), x.end()};
    }

    auto x_end() const -> x_iterator
    {
      return x_iterator{x.end(), x.end()};
    }

    auto y_begin() const -> y_iterator
    {
      return y.begin();
    }

    auto y_end() const -> y_iterator
    {
      return y.end();
    }

    auto data_transform_begin() const -> data_transform_iterator
    {
      return t.begin();
    }

    auto data_transform_end() const -> data_transform_iterator
    {
      return t.end();
    }
  };


  class TransformedImageSegmentationTrainingDataSet
      : public TransformedTrainingDataSet<std::string, std::string,
                                          ImageDataTransform>
  {
    using base_type = TransformedTrainingDataSet<std::string, std::string,
                                                 ImageDataTransform>;

  public:
    using x_iterator = ImageDataSetIterator<Image<Rgb8>>;
    using y_iterator = ImageDataSetIterator<Image<int>>;
    using data_transform_iterator =
        typename data_transform_set_type::const_iterator;
    using iterator =
        TransformedTrainingDataSetIterator<x_iterator, y_iterator,
                                           data_transform_iterator>;

    inline TransformedImageSegmentationTrainingDataSet() = default;

    inline auto begin() const -> iterator
    {
      return iterator{x_begin(), y_begin(), data_transform_begin()};
    }

    inline auto end() const -> iterator
    {
      return iterator{x_end(), y_end(), data_transform_end()};
    }

    auto x_begin() const -> x_iterator
    {
      return x_iterator{x.begin(), x.end()};
    }

    auto x_end() const -> x_iterator
    {
      return x_iterator{x.end(), x.end()};
    }

    auto y_begin() const -> y_iterator
    {
      return y_iterator{y.begin(), y.end()};
    }

    auto y_end() const -> y_iterator
    {
      return y_iterator{y.end(), y.end()};
    }

    auto data_transform_begin() const -> data_transform_iterator
    {
      return t.begin();
    }

    auto data_transform_end() const -> data_transform_iterator
    {
      return t.end();
    }
  };

  DO_SARA_EXPORT
  void read_from_csv(TransformedImageClassificationTrainingDataSet& data_set,
                     const std::string& csv_filepath);

  DO_SARA_EXPORT
  void
  write_to_csv(const TransformedImageClassificationTrainingDataSet& data_set,
               const std::string& csv_filepath);

  DO_SARA_EXPORT
  void read_from_csv(TransformedImageSegmentationTrainingDataSet& data_set,
                     const std::string& csv_filepath);

  DO_SARA_EXPORT
  void write_to_csv(const TransformedImageSegmentationTrainingDataSet& data_set,
                    const std::string& csv_filepath);

} /* namespace Sara */
} /* namespace DO */
