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
