from .models import UserSubmitting

def my_scheduled_job():
    print('this is cron job')
    all_users = UserSubmitting.objects.all()
    for u in all_users:
        print('user:', u, 'submit:', u.submit_times)
        u.submit_times = 10
        u.save()