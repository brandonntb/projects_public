#!/bin/bash

usernames=("bob" "dylan" "aaron")
passwords=("password1" "password2" "password3")

for username in "${usernames[@]}"
do
    # && chains the commands together
    useradd -m -s /bin/bash "$username" && for password in "${passwords[@]}"; do echo "$username:$password" | chpasswd ; done

    echo "User $username created"
done



