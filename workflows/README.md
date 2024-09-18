## Additional Notes

- If you don't want to use two control nets, you should use single_transfer_vx.json workflows

- v1 versions are more stable and generate better results. But they have less control over style transfer.

- v2 versions have more control over style transfering. But depending on the input parameters, tensors could deviate from the control net output distribution. This could result in noisy images. To avoid this, you can use the *_v1 versions of the workflows.