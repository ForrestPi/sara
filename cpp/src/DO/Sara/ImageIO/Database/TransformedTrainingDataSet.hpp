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

  template <typename XIterator, typename YIterator, typename TIterator,
           typename XYTReferenceIterator>
  class TransformedTrainingDataSetIterator
  {
  public:
    using xyt_ref_iterator = XYTReferenceIterator;
    using x_iterator = XIterator;
    using y_iterator = YIterator;
    using data_transform_iterator = TIterator;

    using x_type = typename XIterator::value_type;
    using y_type = typename YIterator::value_type;
    using data_transform_type = typename TIterator::value_type;
    using self_type = TransformedTrainingDataSetIterator;

    inline TransformedTrainingDataSetIterator() = default;

    inline TransformedTrainingDataSetIterator(xyt_ref_iterator xyt_ref_i,
                                              x_iterator x, y_iterator y,
                                              data_transform_iterator t)
      : _xyt_ref_i{xyt_ref_i}
      , _x{x}
      , _y{y}
      , _t{t}
    {
    }

    inline auto operator++() -> self_type&
    {
      ++xyt_ref_i;
      return *this;
    }

    inline auto operator--() -> self_type&
    {
      --xyt_ref_i;
      return *this;
    }

    inline auto operator+=(std::ptrdiff_t n) -> self_type&
    {
      xyt_ref_i += n;
      return *this;
    }

    inline auto operator-=(std::ptrdiff_t n) -> self_type&
    {
      xyt_ref_i -= n;
      return *this;
    }

    inline auto x() const -> x_iterator
    {
      return _x_data + std::get<0>(*_xyt_ref_i);
    }

    inline auto y() const -> y_iterator
    {
      return _y_data + std::get<0>(*_xyt_ref_i);
    }

    inline auto t() const -> data_transform_iterator
    {
      return _t_data + std::get<1>(*_xyt_ref_i);
    }

    inline auto x_ref() const -> const x_type&
    {
      return *x();
    }

    inline auto y_ref() const -> const y_type&
    {
      return *y();
    }

    inline auto t_ref() const -> const data_transform_type&
    {
      return *t();
    }

    inline auto operator==(const self_type& other) const -> bool
    {
      return std::make_tuple(_xyt_ref, _x_data, _y_data, _t_data) ==
             std::make_tuple(other._xyt_ref, other._x_data, other._y_data,
                             other._t_data);
    }

    inline auto operator!=(const self_type& other) const -> bool
    {
      return !(operator==(other));
    }

  private:
    xyt_ref_iterator _xyt_ref_i;

    x_iterator _x_data;
    y_iterator _y_data;
    data_transform_iterator _t_data;
  };


  template <typename XHandle, typename YHandle, typename DataTransform>
  class TransformedTrainingDataSet
  {
  public:
    using x_handle = XHandle;
    using y_handle = YHandle;
    using data_transform_type = DataTransform;

    using x_set_type = std::vector<XHandle>;
    using y_set_type = std::vector<YHandle>;
    using data_transform_set_type = std::vector<DataTransform>;

    using x_pointer = typename x_set_type::const_pointer;
    using y_pointer = typename y_set_type::const_pointer;
    using t_pointer = typename y_set_type::const_pointer;

    inline TransformedTrainingDataSet() = default;

    inline void clear()
    {
      xyt_refs.clear();
      x.clear();
      y.clear();
      t.clear();
    }

    inline bool operator==(const TransformedTrainingDataSet& other) const
    {
      return x == other.x && y == other.y && t == other.t && xyt_refs =
                 other.xyt_refs;
    }

    inline bool operator!=(const TransformedTrainingDataSet& other) const
    {
      return !(*this == other);
    }

    std::vector<std::array<std::size_t, 2>> xyt_refs;

    x_set_type x;
    y_set_type y;
    data_transform_set_type t;
  };


  class TransformedImageClassificationTrainingDataSet
      : public TransformedTrainingDataSet<std::string, int, ImageDataTransform>
  {
    using base_type =
        TransformedTrainingDataSet<std::string, int, ImageDataTransform>;

  public:
    using x_iterator = ImageDataSetIterator<Image<Rgb8>>;
    using y_iterator = typename y_set_type::const_iterator;
    using data_transform_iterator =
        typename data_transform_set_type::const_iterator;
    using iterator =
        TransformedTrainingDataSetIterator<x_iterator, y_iterator,
                                           data_transform_iterator>;

    inline TransformedImageClassificationTrainingDataSet() = default;

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
