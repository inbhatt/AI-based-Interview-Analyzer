from django.contrib import admin
from .models import User, Expression, Eyes, HandsExpression, Speech,CandidateRecord

admin.site.register(User)
admin.site.register(Expression)
admin.site.register(Eyes)
admin.site.register(HandsExpression)
admin.site.register(Speech)
admin.site.register(CandidateRecord)

