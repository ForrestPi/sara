#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO::Sara;

GRAPHICS_MAIN()
{
  // Open a 300x200 window with top-left corner (10,10).
  Window w1 = create_window(300, 200, "A first window", 10, 10);

  // Draw a 150x100 red rectangle with top-left corner at (20, 10).
  draw_rect(20, 10, 150, 100, Red8);

  // Create a second window with dimensions 200x300 with top-left corner (320, 10)
  Window w2 = create_window(200, 300, "A second window", 330, 10);
  // To draw on the second window, we need to tell the computer that we want
  // "activate" it.
  set_active_window(w2);
  // Draw a blue line from coordinates (20, 10) to coordinates (150, 270).
  draw_line(20, 10, 150, 270, Blue8);

  // Draw another green line but on the first window.
  set_active_window(w1);
  draw_line(20, 10, 250, 170, Green8);

  // Wait for a click in any window.
  any_click();

  // It is OK if we forget to close the windows, there will be no memory leak.

  return 0;
}