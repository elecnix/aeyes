This repository is called A-Eyes, A and eyes like human eyes. It sounds like AI's
The goal is to provide AI with eyes. I want you to create a CLI that I can install on my computer and that allows an AI to capture images from a webcam.
the CLI must start a daemon process and return to the user shell. The daemon process should keep the webcam open and the stream should go nowhere except when the CLI is started and then a user requests an image using the CLI
The daemon exists so that the webcam can take a great picture quickly without waiting for auto-exposure and auto-focus.
The program should be written in a language that is suitable for this kind of thing, so probably not Node.js but maybe Rust.
The CLI, when invoked with no arguments or invalid arguments, should always respond with a useful output that can tell AI how to proceed next. It does not need to be too contextual, just print the main help.
This program should work on Windows, Linux and Mac OS, but you will develop it initially on Linux.
You have a webcam available right now, and you can test your own dog food.
There should be tests that create a fake webcam and that verify basic functionality.
There should be unit tests that cover at least 80% of the code base globally.
there should be a github action that runs on every push to a branch
I want you to push this repo to github using the ghcli which I already have logged in.
Please commit early so I can fork the development onto Windows and Mac OS coding agents
You have to decide which video system, library, whatever to use on Linux You may implement one plugin for each
yes the system should be pluggable so it can support multiple video systems
It should provide the highest-possible quality images
it should support autofocus, auto-exposure
the CLI should select a camera by name or ID, never index, so the selection is stable as cameras are added or removed
if there is more than one camera, the CLI should list them and ask the user to select one (non-interactively)
all invocations should be non-interactive, so the CLI does not block an AI agent
the daemon should expose an HTTP API that exposes a simple GET /cams/<id>/frame to grab a frame, and GET /cams to list them
all generated files should be git-ignored
ship it so it can be installed on linux, windows, macos
make it installable with all os-specific tools: brew, apt, winget, etc.
the test coverage should be kept above 80%
the daemon should respond with comprehensive error messages
the daemon should auto-start on the first frame grab request
the daemon should auto-stop after a period of inactivity (configurable, default to 1 hour)
GitHub actions should test linux, windows, macos on every push
The CLI should support opening multiple streams, so the user can request from any webcam with no fuss.
