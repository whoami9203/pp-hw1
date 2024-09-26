#!/bin/bash

# Start the SSH agent in the background
eval "$(ssh-agent -s)"

# Add your SSH private key
ssh-add ~/.ssh/id_rsa_git

# Print a success message
echo "SSH agent started and key added."
