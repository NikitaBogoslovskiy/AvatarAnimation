from progress.bar import Bar


class TrainingBar(Bar):
    suffix = '%(index)d/%(max)d loss = %(current_loss).6f'

    def set_loss(self, loss):
        self.loss = loss

    @property
    def current_loss(self):
        return self.loss