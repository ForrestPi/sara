// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_MATCH_MATCH_HPP
#define DO_MATCH_MATCH_HPP

namespace DO {

  class Match
  {
  public:

    enum MatchingDirection { SourceToTarget, TargetToSource };
    //! Default constructor
    inline Match()
      : x_(0), y_(0)
      , target_rank_(-1), score_(std::numeric_limits<float>::max())
      , matching_dir_(SourceToTarget)
      , x_ind_(-1), y_ind_(-1) {}

    inline Match(const Keypoint *x,
                 const Keypoint *y,
                 float score = std::numeric_limits<float>::max(),
                 MatchingDirection matchingDir = SourceToTarget,
                 int indX = -1, int indY = -1)
      : x_(x), y_(y)
      , target_rank_(-1), score_(score)
      , matching_dir_(matchingDir), x_ind_(indX), y_ind_(indY)
    {}

    //! Constant accessors.
    bool isSKeyNull() const { return x_ == 0; }
    bool isTKeyNull() const { return y_ == 0; }
    const Keypoint& source() const { if (isSKeyNull()) exit(-1); return *x_; }
    const Keypoint& target() const { if (isTKeyNull()) exit(-1); return *y_; }
    const OERegion& sFeat() const { return source().feat(); }
    const OERegion& tFeat() const { return target().feat(); }
    const Point2f& sPos() const { return sFeat().center(); }
    const Point2f& tPos() const { return tFeat().center(); }
    int rank() const { return target_rank_; }
    float score() const { return score_; }
    MatchingDirection matchingDir() const { return matching_dir_; }
    int sInd() const { return x_ind_; }
    int tInd() const { return y_ind_; }
    Vector2i indexPair() const { return Vector2i(x_ind_, y_ind_); }

    //! Non-constant accessors.
    const Keypoint *& sPtr() { return x_; }
    const Keypoint *& tPtr() { return y_; }
    int& rank() { return target_rank_; }
    float& score() { return score_; }
    MatchingDirection& matchingDir() { return matching_dir_; }
    int& sInd() { return x_ind_; }
    int& tInd() { return y_ind_; }

    //! Key match equality.
    bool operator==(const Match& m) const
    { return (source() == m.source() && target() == m.target()); }

  private: /* data members */
    const Keypoint *x_;
    const Keypoint *y_;
    int target_rank_;
    float score_;
    MatchingDirection matching_dir_;
    int x_ind_, y_ind_;
  };

  inline Match indexMatch(int i1, int i2)
  { return Match(0, 0, std::numeric_limits<float>::max(), Match::SourceToTarget, i1, i2); }

  //! I/O
  std::ostream & operator<<(std::ostream & os, const Match& m);

  bool writeMatches(const std::vector<Match>& matches, const std::string& fileName);

  bool readMatches(std::vector<Match>& matches, const std::string& fileName, float scoreT = 10.f);

  bool readMatches(std::vector<Match>& matches,
    const std::vector<Keypoint>& sKeys, const std::vector<Keypoint>& tKeys,
    const std::string& fileName, float scoreT = 10.f);

  //! View matches.
  void drawImPair(const Image<Rgb8>& I1, const Image<Rgb8>& I2, const Point2f& off2, float scale = 1.0f);

  inline void drawImPairH(const Image<Rgb8>& I1, const Image<Rgb8>& I2, float scale = 1.0f)
  { drawImPair(I1, I2, Point2f(I1.width()*scale, 0.f), scale); }
  inline void drawImPairV(const Image<Rgb8>& I1, const Image<Rgb8>& I2, float scale = 1.0f)
  { drawImPair(I1, I2, Point2f(0.f, I1.height()*scale), scale); }

  void drawMatch(const Match& m, const Color3ub& c, const Point2f& off2, float z = 1.f);

  void drawMatches(const std::vector<Match>& matches, const Point2f& off2, float z = 1.f);

  void checkMatches(const Image<Rgb8>& I1, const Image<Rgb8>& I2, 
                    const std::vector<Match>& matches, bool redrawEverytime = false, float z = 1.f);

} /* namespace DO */

#endif /* DO_MATCH_MATCH_HPP */