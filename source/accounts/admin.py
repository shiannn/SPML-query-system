from django.contrib import admin
from .models import UserSubmitting
from .forms import SignUpForm
#from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

class UserAdmin(admin.ModelAdmin):
    add_form = SignUpForm
    model = UserSubmitting
    list_display = ['username', 'submit_times']

    def get_form(self, request, obj=None, **kwargs):
        """
        Use special form during foo creation
        """
        defaults = {}
        if obj is None:
            defaults['form'] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)

admin.site.register(UserSubmitting, UserAdmin)