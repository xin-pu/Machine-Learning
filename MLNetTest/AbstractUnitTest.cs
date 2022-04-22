using MLNet;
using Xunit.Abstractions;

namespace MLNetTest
{
    public class AbstractUnitTest
    {
        public AbstractUnitTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
            Log.print = print;
        }


        internal ITestOutputHelper _testOutputHelper { get; }

        internal void print<T>(T[] array)
        {
            _testOutputHelper.WriteLine(string.Join("\r\n", array));
        }


        internal void print(object? obj)
        {
            _testOutputHelper.WriteLine(obj?.ToString());
        }
    }
}