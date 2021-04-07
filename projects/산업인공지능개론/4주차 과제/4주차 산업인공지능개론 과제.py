# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 23:10:13 2021

@author: 고정재
"""

from durable.lang import *

with ruleset('product'):
    # 규칙 1 : scratch1은 길이가 300um 이상이면 NG 이다.
    @when_all((m.defect == 'scratch1') & (m.length >= 300))
    def result_1(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'NG'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))
    # 규칙 2 : scratch2은 길이가 300um 이하이면 OK 이다.
    @when_all((m.defect == 'scratch2') & (m.length <= 300))
    def result_2(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))
   # 규칙 3 : 뜯김1은 길이가 100um 이하이면 OK 이다.
    @when_all((m.defect == '뜯김1') & (m.length >= 100))
    def result_3(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number)) 
   # 규칙 4 : 뜯김2은 길이가 50um 이하이면 특채출하이다.
    @when_all((m.defect == '뜯김2') & (m.length >= 50))
    def result_4(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : '특채출하'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number)) 
  # 규칙 5 : 수율 90% 이하이면 출고보류이다.
    @when_all((m.defect == '수율') & (m.amount <= 90))
    def result_5(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : '출고보류'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))
   # 규칙 6 : 얼룩은 주기성이면 수율 0% 이므로 NG이다.
    @when_all((m.defect == '얼룩') & (m.amount == 0))
    def result_6(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'NG'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number)) 
    # 규칙 7 : 핀홀은 BLU에서 안 보이면 OK 이다.
    @when_all((m.defect == '핀홀') & (m.육안 == 'OK'))
    def result_7(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number)) 
    # 규칙 8 : 점박이는 Bluelight 에서 안 보이면 OK 이다.
    @when_all((m.defect == '점박이') & (m.bluelight == 'OK'))
    def result_8(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))
    # 규칙 9 : 금형주기성 dimple은 10개 이하이면 OK이다.
    @when_all((m.defect == 'dimple1') & (m.갯수 <= 10))
    def result_9(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))    
    # 규칙 10 : 가이드롤 주기성 dimple은 2개 이하만 OK 이다.
    @when_all((m.defect == 'dimple2') & (m.갯수 <= 2))
    def result_10(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number)) 
    # 규칙 11 : 부착력은 90%이상만 출고 할수 있다.
    @when_all((m.defect == '부착력') & (m.amount >= 90))
    def result_11(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : '출고'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))   
    # 규칙 12 : 에릭슨 테스트는 8mm 이상 되어야 OK 이다.
    @when_all((m.defect == '에릭슨') & (m.두께 >= 8))
    def result_12(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number)) 
    # 규칙 13 : 휘도 값은 100%이상 나와야지 출고 할수 있다.
    @when_all((m.defect == '휘도') & (m.amount >= 100))
    def result_13(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : '출고'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))  
    # 규칙 14 : 글로수값은 30%이상 나와야 OK 이다.
    @when_all((m.defect == '글로수') & (m.amount >= 30))
    def result_14(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))
    # 규칙 15 : 헤이즈는 90%이상 나와야 OK 이다.
    @when_all((m.defect == '헤이즈') & (m.amount >= 90))
    def result_15(c):
        c.assert_fact({'number' : c.m.number, 'defect_name' : c.m.defect, 'defect_result' : 'OK'})
        print('Input Num : {0}, defect Processing.......'.format(c.m.number))
        
    # 결과 출력
    @when_all(m.defect_result == 'NG')
    def print_result_1(c):
        print('Number : {0} , defect : {1} , defect Result : NG product!\n'.format(c.m.number, c.m.defect_name))
    @when_all(m.defect_result == 'OK')
    def print_result_2(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name)) 
    @when_all(m.defect_result == 'OK')
    def print_result_3(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name))    
    @when_all(m.defect_result == '특채출하')
    def print_result_4(c):
        print('Number : {0} , defect : {1} , defect Result : 특채출하 product!\n'.format(c.m.number, c.m.defect_name))
    @when_all(m.defect_result == '출고보류')
    def print_result_5(c):
        print('Number : {0} , defect : {1} , defect Result : 출고보류 product!\n'.format(c.m.number, c.m.defect_name)) 
    @when_all(m.defect_result == 'NG')
    def print_result_6(c):
        print('Number : {0} , defect : {1} , defect Result : NG product!\n'.format(c.m.number, c.m.defect_name))  
    @when_all(m.defect_result == 'OK')
    def print_result_7(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name)) 
    @when_all(m.defect_result == 'OK')
    def print_result_8(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name))  
    @when_all(m.defect_result == 'OK')
    def print_result_9(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name))  
    @when_all(m.defect_result == 'OK')
    def print_result_10(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name))
    @when_all(m.defect_result == '출고')
    def print_result_11(c):
        print('Number : {0} , defect : {1} , defect Result : 출고 product!\n'.format(c.m.number, c.m.defect_name)) 
    @when_all(m.defect_result == 'OK')
    def print_result_12(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name)) 
    @when_all(m.defect_result == '출고')
    def print_result_13(c):
        print('Number : {0} , defect : {1} , defect Result : 출고 product!\n'.format(c.m.number, c.m.defect_name)) 
    @when_all(m.defect_result == 'OK')
    def print_result_14(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name))   
    @when_all(m.defect_result == 'OK')
    def print_result_15(c):
        print('Number : {0} , defect : {1} , defect Result : OK product!\n'.format(c.m.number, c.m.defect_name))   
         
#scratch  데이터 입력
assert_fact('product', { 'number': '1', 'defect': 'scratch1', 'length' : 300})
assert_fact('product', { 'number': '2', 'defect': 'scratch2', 'length' : 300})
#뜯김 데이터 입력
assert_fact('product', { 'number': '3', 'defect': '뜯김1', 'length' : 100})
assert_fact('product', { 'number': '4', 'defect': '뜯김2', 'length' : 50})
#수율 데이터 입력
assert_fact('product', { 'number': '5', 'defect': '수율', 'amount' : 90})
#얼룩 데이터 입력
assert_fact('product', { 'number': '6', 'defect': '얼룩', 'amount' : 0})
#핀홀 데이터 입력
assert_fact('product', { 'number': '7', 'defect': '핀홀', '육안' : 'OK'})
#점박이 데이터 입력
assert_fact('product', { 'number': '8', 'defect': '점박이', 'bluelight' : 'OK'})
#dimple 데이터 입력
assert_fact('product', { 'number': '9', 'defect': 'dimple1', '갯수' : 10})
assert_fact('product', { 'number': '10', 'defect': 'dimple2', '갯수' : 2})
#부착력 데이터 입력
assert_fact('product', { 'number': '11', 'defect': '부착력', 'amount' : 100})
#에릭슨 데이터 입력
assert_fact('product', { 'number': '12', 'defect': '에릭슨', '두께' : 8})
#휘도 데이터 입력
assert_fact('product', { 'number': '13', 'defect': '휘도', 'amount' : 100})
#글로수 데이터 입력
assert_fact('product', { 'number': '14', 'defect': '글로수', 'amount' : 30})
#헤이즈 데이터 입력
assert_fact('product', { 'number': '15', 'defect': '헤이즈', 'amount' : 90})








                        

                        












