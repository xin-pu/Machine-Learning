using System.Text;
using AutoDiff;
using MLNet.Kernels;
using MLNet.LearningModel;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Optimizers;
using MLNet.Utils;
using Numpy;
using YAXLib;
using YAXLib.Attributes;

namespace MLNet.Models
{
    /// <summary>
    ///     Learning Model
    /// </summary>
    public abstract class Model : ViewModelBase
    {
        protected Model()
        {
            Name = GetType().Name;
        }


        public string Name { protected get; set; }
        public string FilePath => $"{Name}.xml";
        public bool Print { set; get; } = true;


        [YAXDontSerialize] public NDarray Resolve { set; get; } = null!;

        [YAXDontSerialize] public Kernel Kernel { set; get; } = null!;

        [YAXDontSerialize] public Loss CostFunc { protected set; get; } = null!;

        [YAXDontSerialize] public Metric[] Metrics { protected set; get; } = { };

        [YAXDontSerialize] public Optimizer Optimizer { protected set; get; } = null!;

        [YAXDontSerialize] public Variable[] Variables { protected set; get; } = { };


        /// <summary>
        ///     Configures the model for training.
        /// </summary>
        /// <param name="optimizer"></param>
        /// <param name="loss"></param>
        /// <param name="metric"></param>
        public void Compile(Optimizer optimizer, Loss loss, Metric[] metric)
        {
            Optimizer = optimizer;
            CostFunc = loss;
            Metrics = metric;
        }

        public NDarray Call(NDarray x)
        {
            return call(x);
        }

        public NDarray Predict(NDarray x)
        {
            /// Step 1 Transform
            var x_cvt = transform(x);

            /// Step 2 Casll 
            var y_pred = call(x_cvt);

            return y_pred;
        }

        public void Evaluate(NDarray x, NDarray y)
        {
            var y_pred = Predict(x);

            Metrics.AsParallel().ForAll(m =>
            {
                m.Call(y, y_pred);
                print(m);
            });
        }

        public void Fit(
            NDarray traindatas_x,
            NDarray trandatas_y,
            TrainConfig trainConfig)
        {
            try
            {
                print($"{Name} Start Fit:\r\n");

                /// Step 1 Convert Model
                var x_cvt = transform(traindatas_x);

                /// Step 2 Create Loss Function
                var featureCount = x_cvt.shape[1];
                var w = Enumerable.Range(0, featureCount).Select(_ => new Variable()).ToArray();
                CostFunc = initialLoss(w, x_cvt, trandatas_y);

                /// Step 3 Fit

                var resolveTemp = np.random.randn(featureCount, 1);

                Enumerable.Range(0, trainConfig.Epoch).ToList().ForEach(e =>
                {
                    var (gradarray, loss) = CostFunc.Call(resolveTemp);

                    var grad = np.expand_dims(np.array(gradarray), -1);
                    resolveTemp -= trainConfig.LearningRate * grad;

                    if (Print)
                        Log.print?.Invoke($"Epoch:\t{e:D4}\tLoss:{loss:F4}");
                });
                Resolve = resolveTemp;

                /// Step 4 Evalate
                Evaluate(traindatas_x, trandatas_y);
            }
            catch (Exception ex)
            {
                print($"{Name} Predict:\r\n{ex.Message}");
            }
        }


        public override string ToString()
        {
            var strBuild = new StringBuilder();
            strBuild.AppendLine($"{Name}");
            strBuild.AppendLine($"Resolve:\r\n{Resolve}");
            return strBuild.ToString();
        }


        #region print

        internal void print(object obj)
        {
            if (Print)
                Log.print?.Invoke(obj);
        }

        public void PrintSelf()
        {
            Log.print?.Invoke(this);
        }

        internal void printTitle(string title)
        {
        }

        #endregion

        #region internal

        internal virtual NDarray transform(NDarray x)
        {
            if (Kernel == null)
                return transformer.to_linear_firstorder(x);

            var kernel = Kernel.Transform(x);
            var final = transformer.to_linear_firstorder(kernel);
            return final;
        }


        internal abstract NDarray call(NDarray x);

        internal abstract Loss initialLoss(Variable[] variables, NDarray x, NDarray y);

        #endregion

        #region serialize

        public virtual void Save(string path)
        {
            using var stream = File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var textWriter = new StreamWriter(stream);
            var serializer = new YAXSerializer(typeof(Model));
            serializer.Serialize(this, textWriter);
            Resolve.tofile(FilePath, "", "");
            textWriter.Flush();
        }

        public static Model Load(string path)
        {
            using var stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var textReader = new StreamReader(stream);
            var type = typeof(Model);
            var deserializer = new YAXSerializer(type);
            var model = (Model) deserializer.Deserialize(textReader);
            var r = np.fromfile(model.FilePath);
            model.Resolve = np.expand_dims(r, -1);
            return model;
        }

        #endregion
    }
}