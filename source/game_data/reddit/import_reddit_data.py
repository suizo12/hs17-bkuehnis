from schedule.models import NbaReddit

import praw
import datetime
import time
from datetime import timedelta


def import_reddit_game_thread():
    reddit = praw.Reddit(client_id='V9_C9u8ECiN8yg',
                         client_secret='8GDM7lfEpseZzNcuUmLuZ-Zx368',
                         password='9ihaddAjcSpVf2jU',
                         user_agent='testscript by /u/hsr_nsfs',
                         username='hsr_nsfs')

    today = datetime.date.today()
    a_week_ago = today - timedelta(days=2)

    start_dt = datetime.datetime(year=a_week_ago.year, month=a_week_ago.month, day=a_week_ago.day, hour=0).timetuple()
    today = datetime.date.today() + timedelta(days=1)
    end_dt = datetime.datetime(year=today.year, month=today.month, day=today.day, hour=0).timetuple()

    s_t = time.mktime(start_dt)
    e_t = time.mktime(end_dt)

    #below is something like what I want
    #search_str = "title contains:'XXXXX' and (timestamp:" + str(int(s_t)) + ".." + str(int(e_t)) + ")"

    #below is what works for just the timestamp

    #thread_array = reddit.subreddit('nba').search(search_str, subreddit="nba", limit=1000, syntax="cloudsearch")
    # extra_query="flair_text:'Post Game Thread'"

    # clean up data
    # remove_no_post_game_threads()

    for submission in reddit.subreddit('nba').submissions(int(s_t), int(e_t), extra_query="title:'*Post Game Thread*'"):
        nba_reddit = NbaReddit.objects.filter(name__exact=submission.name)
        print(submission.name)
        print(submission.title)
        if not nba_reddit.exists():
            if 'post game thread' in submission.title.lower() or 'post-game thread' in submission.title.lower():
                new_nba_reddit = NbaReddit(title=submission.title, ups=submission.ups, permalink=submission.permalink, name=submission.name, created = submission.created_utc)
                new_nba_reddit.save()
        else:
            nba_reddit = nba_reddit.first()
            nba_reddit.ups = submission.ups
            nba_reddit.save()


def remove_no_post_game_threads():
    for r in NbaReddit.objects.all():
        if 'Next day thread' in r.title:
            r.delete()
        if not 'post game thread' in r.title.lower():
            if not 'post-game thread' in r.title.lower():
                r.delete()
