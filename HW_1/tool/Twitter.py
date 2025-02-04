import re
import time
from datetime import datetime, timedelta

url_pattern = re.compile(r"http\S+")
hashtag_pattern = re.compile(r"#\S+")
mention_pattern = re.compile(r"@\S+")
rule = re.compile(r"[^a-zA-Z0-9\s]")
# rule = re.compile(r'[^a-zA-Z\s]')
rule_all = re.compile(r"[^a-zA-Z0-9\s]")


class User:
    """
    TwitterUser class can used to save a user
    Including userbased features
    """

    # user_feature = []
    def __init__(self, tweet_json):
        self.user_json = tweet_json["user"]
        self.id_str = self.user_json["id_str"]
        self.screen_name = self.user_json["screen_name"]
        self.name = self.user_json["name"]
        self.created_at = self.json_date_to_stamp(self.user_json["created_at"])
        self.followers_count = self.get_followers_count()
        self.friends_count = self.get_friends_count()
        self.statuses_count = self.get_statuses_count()

    def json_date_to_stamp(self, json_date):
        """
        exchange date from json format to timestamp(int)
        input:
            date from json
        output:
            int
        """
        time_strpt = time.strptime(json_date, "%a %b %d %H:%M:%S +0000 %Y")
        stamp = int(time.mktime(time_strpt))
        return stamp

    def json_date_to_os(self, json_date):
        """
        exchange date from json format to linux OS format
        input:
            date from json
        output:
            datetime
        """
        time_strpt = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.strptime(json_date, "%a %b %d %H:%M:%S +0000 %Y"),
        )
        os_time = datetime.strptime(str(time_strpt), "%Y-%m-%d %H:%M:%S")
        return os_time

    def get_followers_count(self):
        """
        return follower count
        """
        followers_count = self.user_json["followers_count"]
        if followers_count == 0:
            followers_count = 1
        return followers_count

    def get_friends_count(self):
        """
        return friends count
        """
        friends_count = self.user_json["friends_count"]
        if friends_count == 0:
            friends_count = 1
        return friends_count

    def get_statuses_count(self):
        """
        return statuses count
        """
        statuses_count = self.user_json["statuses_count"]
        if statuses_count == 0:
            statuses_count = 1
        return statuses_count

    def get_user_age(self):
        """
        Age of an account
        get age feature of an account, remember call this function all the time. Time exchange
        """
        account_start_time = self.json_date_to_os(self.user_json["created_at"])
        now_time = datetime.now()
        account_age = (now_time - account_start_time).days
        if account_age == 0:
            account_age = 1
        return account_age

    def get_user_favourites(self):
        """
        get user favourites count
        """
        favourites_count = self.user_json["favourites_count"]
        return favourites_count

    def get_user_name(self):
        """
        get user name
        """
        name = self.user_json["screen_name"]
        return name
    
    def get_listed_count(self):
        count = self.user_json["listed_count"]
        return count


class Tweet:
    """
    TwitterTweet class can used to save a tweet
    """

    def __init__(self, tweet_json):
        self.tweet_json = tweet_json
        self.user = User(tweet_json)

        self.text = tweet_json["text"]
        self.timestr = self.json_date_to_stamp(tweet_json["created_at"])
        self.tid_str = self.tweet_json["id_str"]

    def json_date_to_stamp(self, json_date):
        """
        exchange date from json format to timestamp(int)
        input:
            date from json
        output:
            int
        """
        time_strpt = time.strptime(json_date, "%a %b %d %H:%M:%S +0000 %Y")
        stamp = int(time.mktime(time_strpt))
        return stamp

    def is_en(self):
        return self.tweet_json["lang"] == "en"

    def get_id(self):
        return self.tid_str

    def get_hashtag_count(self):
        """
        get number of hashtags
        """
        hashtags = self.tweet_json["entities"]["hashtags"]
        return len(hashtags)

    def get_user_mentions(self):
        """
        get number of user_mentions
        """
        mentions = self.tweet_json["entities"]["user_mentions"]
        return len(mentions)

    def get_text_len(self):
        """
        return chars of text
        """
        return len(self.text)

    def get_contributors(self):
        '''
        get contributers count
        '''
        contributors = self.tweet_json["contributors"]
        return 1 if contributors == None else contributors

    def get_quote_count(self):
        '''
        get the quote count
        '''
        rt_quote = self.tweet_json["quote_count"]
        return rt_quote
    
    def get_lang(self):
        '''
        get the language
        '''
        lang = self.tweet_json["lang"]
        return lang
    
    def get_retweet_self_count(self):
        return self.tweet_json["retweet_count"]

    def get_tweet_features(self):
        feature_list = []
        feature_list.append(self.user.get_user_age())  # 1
        feature_list.append(self.user.get_user_favourites())  # 5
        feature_list.append(self.get_hashtag_count())  # 8
        feature_list.append(self.get_text_len())  # 11

        # include 8 more new features in here and return the list of features
        feature_list.append(self.user.get_followers_count())  # 1 get the followers count
        feature_list.append(self.user.get_friends_count())  # 2 get the friends count
        feature_list.append(self.user.get_statuses_count())  # 3 get the statuses count
        feature_list.append(self.get_user_mentions())  # 4 get the user mentions
        feature_list.append(self.get_contributors())  # 5 get the contibutors
        feature_list.append(self.get_quote_count())  # 6 get the quote count
        feature_list.append(self.user.get_listed_count())  # 7 get the tweet language
        feature_list.append(self.get_retweet_self_count())  # 8 get the user's name

        return feature_list
