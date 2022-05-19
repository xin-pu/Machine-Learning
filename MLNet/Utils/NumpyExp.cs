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
            return 1 / (1 + np.exp(-w * a));
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

        /// <summary>
        ///     load txt data with format
        ///     f1,f2,...,fn,label
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public static Tuple<NDarray, NDarray, Dictionary<int, string>> loadClassifyData(string filePath)
        {
            var lines = File.ReadAllLines(filePath).Where(l => l != "");
            var ldata = lines
                .Select(l => l.Split(',', ';')
                    .Where(c => c != "")
                    .ToArray())
                .ToList();
            var height = ldata.Count();
            var width = ldata.Select(l => l.Length).Max();

            var res = ldata.Select(a => a.ToList().Take(width - 1).Select(double.Parse).ToArray()).ToList();
            var x = np.array(new double[height, width - 1]);
            foreach (var i in Enumerable.Range(0, height)) x[i] = np.array(res[i]);

            var y_label = ldata.Select(a => a.Last()).ToList();
            var classes = y_label.Distinct().ToList();
            var dict = classes
                .Select((key, index) => new KeyValuePair<int, string>(index, key))
                .ToDictionary(p => p.Key, p => p.Value);
            var dictR = dict.ToDictionary(p => p.Value, p => p.Key);
            var res2 = y_label.Select(l => dictR[l]).ToArray();
            var y = np.expand_dims(np.array(res2), -1);

            return new Tuple<NDarray, NDarray, Dictionary<int, string>>(x, y, dict);
        }
    }
}