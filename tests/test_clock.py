from medisense.clock import VideoClock, WallClock


def test_wall_clock_advances_with_real_time():
    c = WallClock()
    assert c.advance() > 0
    assert c.now() > 0


def test_video_clock_follows_container_timestamps():
    c = VideoClock(fps=25.0)
    assert c.advance(1000.0) == 1.0
    assert c.advance(5000.0) == 5.0
    assert c.now() == 5.0


def test_video_clock_falls_back_to_frame_count():
    c = VideoClock(fps=10.0)
    for _ in range(10):
        c.advance(None)
    assert abs(c.now() - 1.0) < 1e-9


def test_video_clock_is_monotonic_across_loop_restart():
    """A rewound file must not push detector timestamps into the future."""
    c = VideoClock(fps=25.0)
    c.advance(4000.0)
    c.advance(8000.0)
    at_end = c.now()
    c.advance(0.0)          # loop restart reports no position
    c.advance(1000.0)       # then restarts its timeline
    assert c.now() >= at_end


def test_video_clock_handles_missing_fps():
    c = VideoClock(fps=0.0)
    c.advance(None)
    assert c.now() > 0
