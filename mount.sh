sudo mkdir /mnt/waterchallenge 
if [ ! -d "/etc/smbcredentials" ]; then
    sudo mkdir /etc/smbcredentials
fi
if [ ! -f "/etc/smbcredentials/agaidchallengesdata2023.cred" ]; then 

    sudo bash -c 'echo "username=agaidchallengesdata2023" >> /etc/smbcredentials/agaidchallengesdata2023.cred'
    sudo bash -c 'echo "password=U7wlLZLTOUJA+0F+Z1Q0pfVRYDuChtkwMPDN/MioYPmpxXRSyDj1qff+ls+CJjC2h+Zix0dZ1lZM+AStamIygw==" >>/etc/smbcredentials/agaidchallengesdata2023.cred'
fi
    
sudo chmod 600 /etc/smbcredentials/agaidchallengesdata2023.cred

sudo bash -c 'echo "//agaidchallengesdata2023.file.core.windows.net/waterchallenge /mnt/waterchallenge cifs nofail,credentials=/etc/smbcredentials/agaidchallengesdata2023.cred,dir_mode=0777,file_mode=0777,serverino,nosharesock,actimeo=30">> /etc/fstab'

sudo mount -t cifs //agaidchallengesdata2023.file.core.windows.net/waterchallenge /mnt/waterchallenge -o credentials=/etc/smbcredentials/agaidchallengesdata2023.cred,dir_mode=0777,file_mode=0777,serverino,nosharesock,actimeo=30 