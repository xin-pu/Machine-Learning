using System.Text;

namespace MLNet
{
    public struct TrainConfig
    {
        public TrainConfig(
            int epoch = 100,
            int batch = 0,
            bool shuffle = true)
        {
            Epoch = epoch;
            Batch = batch;
            Shuffle = shuffle;
        }

        public int Epoch { get; set; }
        public int Batch { set; get; }

        public bool Shuffle { set; get; }

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"Epoch:\t{Epoch}\rBatch:\t{Batch}\rLR:");
            return str.ToString();
        }
    }
}