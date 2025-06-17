from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from utils.writer import Writer


def run_test(epoch=-1, name=""):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()

    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        # 调用模型的 test() 方法，返回正确预测的数量和样本总数。
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
