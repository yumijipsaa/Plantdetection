# mmdetection/custom_hooks/validation_loss_hook.py
from mmengine.hooks import Hook
import torch

class ValidationLossHook(Hook):
    def after_train_epoch(self, runner):
        model = runner.model
        model.eval()
        val_loss = 0
        num_batches = 0
        num_samples = 0

        runner.logger.info('Calculating validation loss...')

        for data_batch in runner.val_dataloader:
            with torch.no_grad():
                # forward
                losses = model.loss(data_batch, mode='val')
                # losses는 dict 형태
                batch_loss = sum([v.item() for v in losses.values()])
                val_loss += batch_loss
                num_batches += 1
                num_samples += len(data_batch['inputs'])

        avg_loss = val_loss / num_batches
        runner.logger.info(f'Validation Loss: {avg_loss:.4f}')
        runner.train()  # 다시 train 모드로 전환
