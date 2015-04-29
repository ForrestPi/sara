#include <DO/Graphics.hpp>
#include <DO/VideoIO.hpp>


GRAPHICS_MAIN()
{
  using namespace std;
  using namespace DO;

  const string video_filepath = src_path("orion_1.mpg");

  VideoStream video_stream(video_filepath);
  Image<Rgb8> video_frame;

  while (true)
  {
    video_stream >> video_frame;
    if (!active_window())
      create_window(video_frame.sizes());

    if (!video_frame.data())
      break;
    display(video_frame);
  }

  return 0;
}
