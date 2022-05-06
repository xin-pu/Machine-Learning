namespace MLNet.LearningModel
{
    public struct TrainConfig
    {
        public TrainConfig(
            int epoch = 100,
            int batch = 0,
            double learningRate = 1E-4)
        {
            Epoch = epoch;
            Batch = batch;
            LearningRate = learningRate;
        }

        public int Epoch { get; set; }
        public int Batch { set; get; }
        public double LearningRate { set; get; }
    }
}