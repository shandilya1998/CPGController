!nvidia-smi
!pip install colab_ssh --upgrade
from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
launch_ssh_cloudflared(password="Princyy-12345")
init_git_cloudflared('https://github.com/shandilya1998/CPGController')
!nvidia-smi
