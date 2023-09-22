$usernames = @("barry", "adam", "micheal")
$passwords = @("password1", "password2", "password3")


foreach ($user in $usernames)
{
    $password = $passwords[$usernames.IndexOf($username)]
    New-LocalUser -Name $username -FullName $username -Password $password
    "User $username created"
}
