# coding=UTF-8
import math

from selenium import webdriver
import time
from pymysql.converters import escape_string
from selenium.webdriver.common.keys import Keys
import os
import pymysql

ch_options = webdriver.ChromeOptions()
# ch_options.add_argument('--no-sandbox')
# ch_options.add_argument('--disable-dev-shm-usage')
# ch_options.add_argument('--headless')
# ch_options.add_argument('blink-settings=imagesEnabled=false')
# chrome_path = "./chromedriver"
chrome_path = "./chromedriver.exe"
wd = webdriver.Chrome(executable_path=chrome_path,options=ch_options)

wd.implicitly_wait(3)

# sql_connection = pymysql.connect(host='127.0.0.1', user='root', password='123456',
#                                  db='douban', port=3306, autocommit=False, charset='utf8mb4')

sql_connection = pymysql.connect(host='10.21.11.12', user='root', password='XiaoYuan@0423WenWen1231',
                                 db='douban2', port=3308, autocommit=False, charset='utf8mb4')

cursor = sql_connection.cursor()

def deal_with_page_source(link, username):
    wd.get(link)
    if str(wd.page_source).find("该用户处于锁定或停用状态，无法查看主页内容") != -1:
        try:
            # 更新
            updateUserSql = "UPDATE douban_user SET is_logoff = '%s' Where username = '%s'" % ("1", username)
            print(updateUserSql)
            cursor.execute(updateUserSql)
            sql_connection.commit()
        except Exception as e:
            print(e)
    else:
        try:
            if str(wd.page_source).find('rev-link') != -1:
                follower_count = str(wd.find_element_by_xpath('//p[@class="rev-link"]').text).split("人关注")[0].split("被")[-1]
            else:
                follower_count = 0

            if str(wd.page_source).find('target="_self">成员') != -1:
                following_count = str(wd.find_element_by_xpath('//div[@id="friend"]//span[@class="pl"]').text).split("成员")[-1].split(" )")[0]
            else:
                following_count = 0

            if str(wd.page_source).find('class="user-verify pl"') != -1:
                identity = escape_string(str(wd.find_element_by_xpath('//div[@class="user-verify pl"]').text))
            else:
                identity = ""

        except Exception as e:
            print(e)
            print("获取用户" + username + "基础信息失败")

        if follower_count == 0 and following_count == 0:
            return
        try:
            # 更新
            updateUserSql = "UPDATE douban_user SET identity = '%s', follower_count = '%s',following_count = '%s' Where username = '%s'" % (identity, follower_count, following_count, username)
            print(updateUserSql)
            cursor.execute(updateUserSql)
            sql_connection.commit()
        except Exception as e:
            print(e)
    pass

def login():
    #等待30s用于登录
    wd.get("https://accounts.douban.com/passport/login")
    time.sleep(10)
    # wd.find_element_by_xpath('//li[@class="account-tab-account"]').click()
    # wd.find_element_by_xpath('//input[@class="account-form-input"]').send_keys("15381276737")
    # pwd = wd.find_element_by_xpath('//input[@class="account-form-input password"]')
    # pwd.send_keys("ZZJ747266823")
    # pwd.send_keys(Keys.ENTER)
    # time.sleep(1)

if __name__ == '__main__':
    login()

    selectSql = "SELECT * FROM douban_user WHERE follower_count = 0 and following_count = 0;"
    cursor.execute(selectSql)
    result = cursor.fetchall()

    for res in result[58000:]:
        print(res)
        time.sleep(3)
        # print(res[4])
        deal_with_page_source(res[4], res[0])
        # scarpy_relation(res[4], res[0])