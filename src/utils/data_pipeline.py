import torch
from torch.utils.data import IterableDataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class StreamingDataset(IterableDataset):
    def __init__(self, data_urls, transform=None):
        self.data_urls = data_urls
        self.transform = transform
        
    def __iter__(self):
        # In a real implementation, this would use webdataset to stream tar files
        # from S3 or local disk.
        # For now, we simulate a stream of data.
        logger.info(f"Streaming data from {len(self.data_urls)} sources...")
        
        for url in self.data_urls:
            # Simulate reading a shard
            for i in range(10): # 10 samples per shard
                # Mock image tensor and label
                image = torch.randn(3, 224, 224)
                label = torch.randint(0, 2, (1,)).item()
                
                if self.transform:
                    image = self.transform(image)
                    
                yield image, label

def get_streaming_loader(data_urls, batch_size=32, num_workers=4):
    """
    Returns a DataLoader for streaming data.
    """
    dataset = StreamingDataset(data_urls)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

# Example usage
if __name__ == "__main__":
    urls = ["shard_001.tar", "shard_002.tar"]
    loader = get_streaming_loader(urls, batch_size=4, num_workers=0)
    
    for images, labels in loader:
        print(f"Batch shape: {images.shape}, Labels: {labels}")
        break
