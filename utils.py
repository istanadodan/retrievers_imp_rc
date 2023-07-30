import inspect
from functools import wraps
import logging

def print_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func if isinstance(func, str) else func.__name__

        def print_function_arguments():
            # 함수의 인자 목록 가져오기
            signature = inspect.signature(func)
            parameters = signature.parameters            

            # 위치 인자 값 출력
            arg_prints =''
            for arg_name, arg_value in zip(parameters, args):
                _str = str(arg_value)                
                _out = f"{arg_name}={_str[:min(len(_str), 500)]}]"
                arg_prints += _out +',\n'
            if arg_prints:
                logging.debug(f"### [INPUT: {func_name}] \n{arg_prints[:-2]} ###")

            # 키워드 인자 값 출력
            arg_prints =''
            for arg_name, arg_value in kwargs.items():
                _str = str(arg_value)                
                _out = f"{arg_name}={_str[:min(len(_str), 200)]}]"
                arg_prints += _out +',\n'
            if arg_prints:
                logging.debug(f"### [INPUT: {func_name}] \n{arg_prints[:-2]} ###")
                
        #before
        print_function_arguments()
        #body
        output = func(*args, **kwargs)
        #after
        if isinstance(output, str):
            logging.debug(f"### [OUTPUT: {func_name}] output=\n{output} ###")
        return output
    return wrapper
