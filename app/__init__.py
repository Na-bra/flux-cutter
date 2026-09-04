"""FluxCutter: find someone in a video, cut every appearance into one clip."""

# The one place the version is written down. The bundle's
# CFBundleShortVersionString reads it (packaging/FluxCutter.spec) rather
# than repeating it, because when it was repeated it went stale: the spec
# still said 1.0.0 while v1.1.0 shipped.
#
# Keep it in step with the git tag. tests/test_version.py checks the shape,
# not the value -- only a person can decide whether a change is a patch or
# a minor.
__version__ = "1.5.3"
