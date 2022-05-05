using Numpy;

namespace MLNet.Utils
{
    /// <summary>
    ///     This is a lib extent Numpy
    /// </summary>
    public static class np2
    {
        /// <summary>
        /// </summary>
        /// <param name="a"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public static NDarray sigmoid(NDarray a, double w = 1)
        {
            return 1.0 / (1.0 + np.exp(-w * a));
        }

        public static double variance(NDarray a)
        {
            var mean = a.GetData<double>().Average();

            var delta = np.subtract(a, np.array(mean));

            var varuance = np.power(delta, np.array(2))
                .GetData<double>()
                .Average();

            return varuance;
        }


        public static NDarray load(string filePath)
        {
            var lines = File.ReadAllLines(filePath).Where(l => l != "");
            var ldata = lines
                .Select(l => l.Split(',', ';')
                    .Where(c => c != "")
                    .Select(double.Parse)
                    .ToArray())
                .ToList();
            var height = ldata.Count();
            var width = ldata.Select(a => a.Count()).Max();
            var data = np.array(new double[height, width]);

            foreach (var i in Enumerable.Range(0, height)) data[i] = np.array(ldata[i]);
            return data;
        }
    }
}