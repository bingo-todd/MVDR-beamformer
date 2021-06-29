import basic_tools
import sys

if __name__ == '__main__':

    if len(sys.argv)>1:
        module_name = sys.argv[1]

        if len(sys.argv)>2:
            opt = sys.argv[2]

        if not hasattr(basic_tools,module_name):
            print('incorrect module name')

        if opt == 'test':
            tar_module = getattr(basic_tools,module_name)
            dir(tar_module)
            test_f = getattr(tar_module,'test')
            test_f()
