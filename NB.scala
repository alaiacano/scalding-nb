import com.twitter.scalding._
import cascading.pipe.Pipe
import cascading.flow.FlowDef

class NBTestJob(args: Args) extends Job(args) {
  val input = args("input")
  val output = args("output")

  val iris = Tsv(input, ('id, 'class, 'sepalLength, 'sepalWidth, 'petalLength, 'petalWidth))
    .read

  val irisMelted = iris
    .unpivot(('sepalLength, 'sepalWidth, 'petalLength, 'petalWidth) -> ('feature, 'score))

  val irisTrain = irisMelted.filter('id){id: Int => (id % 3) != 0}.discard('id)

  val irisTest = irisMelted
    .filter('id){id: Int => (id % 3) ==0}
    .discard('class)

  val model = GaussianNB.train(irisTrain)

  val predictions = GaussianNB.classify(irisTest, model).rename(('id, 'class) -> ('id2, 'classPred))

  val results = iris
    .leftJoinWithTiny('id -> 'id2, predictions)
    .discard('id2)
    .map('classPred -> 'classPred) {x: String => Option(x).getOrElse("")}
    .project('id, 'class, 'classPred, 'sepalLength, 'sepalWidth)
    .write(Tsv(output))

}


abstract trait NBCore {
  import Dsl._
  /**
   * Abstract method that must be overwritten to build a model for the distribution-specific
   * model.
   *
   */
  def train(pipe : Pipe, nReducers : Int = 100)(implicit fd: FlowDef) : Pipe

  /**
   * Abstract method that must be overwritten with the distribution-specific
   * implementation.
   *
   * The value that must be returned is <code>Pr({feature set} | class)</code>
   */
  def _joint_log_likelihood(joined : Pipe)(implicit fd: FlowDef) : Pipe

  /**
  * Classification method for Gaussian Naive Bayes.
  *
  * @param data Pipe containing the data to be classified
  * @param model Pipe that was returned from the `train` method.
  * @return A Pipe with the fields `id`, `class` (predicted), and `logLikelihood`.
  */
  def classify(data : Pipe, model : Pipe, nReducers : Int = 100)(implicit fd: FlowDef) = {
    val joined = data
      .skewJoinWithSmaller('feature -> 'feature, model, reducers=nReducers)

    val result = _joint_log_likelihood(joined)
      .mapTo(('id, 'class, 'classPrior, 'sumEvidence) -> ('id, 'class, 'logLikelihood)) {
        values : (String, String, Double, Double) =>
        val (id, className, classPrior, sumEvidence) = values
        (id, className, classPrior + sumEvidence)
      }
      .groupBy('id) {
        _.sortBy('logLikelihood)
         .reverse
         .take(1)
         .reducers(nReducers)
      }
    result
  }

  /**
   * Calculates the prior value for all classes, `Pr(class = C)`
   */
  def classPrior(pipe : Pipe, nReducers : Int = 50)(implicit fd: FlowDef) : Pipe = {
    val counts = pipe.groupBy('class) { _.size('classCount).reducers(nReducers) }
    val totSum = counts.groupAll(_.sum('classCount -> 'totalCount))

    counts.crossWithTiny(totSum)
      .mapTo(('class, 'classCount, 'totalCount) -> ('class, 'classPrior, 'classCount)) {
        x : (String, Double, Double) => (x._1, math.log(x._2 / x._3), x._2)
      }
  }

}

object GaussianNB extends NBCore {
  import Dsl._
  /**
   * Trains a Gaussian Naive Bayes model on the input data.
   *
   * The input `Pipe` must have the fields:
   * <ul><li>class</li><li>feature</li><li>score</li></ul>
   *
   * The output is a pipe fitting the standard NaiveBayes model that we use
   * for all classifiers:
   * <ul>
   * <li>class - the id of the class</li>
   * <li>feature - the id of the feature</li>
   * <li>theta - mean value of the score in each feature/class pair</li>
   * <li>sigma - variance of the score in each feature/class pair</li>
   * <li>classPrior - prior probability of seeing the class</li>
   * </ul>
   *
   * The model parameters calculated are consistent with Scikit-Learn demos.
   */
  def train(pipe : Pipe, nReducers : Int = 100)(implicit fd: FlowDef) : Pipe = {

    val prClass = classPrior(pipe, nReducers).discard('classCount)
    val prFeatureClass = featureStats(pipe, nReducers)

    val model = pipe
      .joinWithSmaller('class -> 'class, prClass, reducers=nReducers)
      .joinWithSmaller(('class, 'feature) -> ('class, 'feature), prFeatureClass, reducers=nReducers)

      .mapTo(('class, 'classPrior, 'feature, 'featureClassSize, 'theta, 'sigma) ->
             ('class, 'feature, 'classPrior, 'theta, 'sigma)) {
        values : (String, Double, String, Double, Double, Double) =>
        val (classId, classPrior, feature, featureClassSize, theta, sigma) = values
        (classId, feature, classPrior, theta, math.pow(sigma, 2))
      }

    model
  }

  def _joint_log_likelihood(joined : Pipe)(implicit fd: FlowDef) : Pipe = {
    val ret = joined
      .map(('theta, 'sigma, 'score) -> 'evidence) {
        values : (Double, Double, Double) => _gaussian_prob(values._1, values._2, values._3)}
      .project('id, 'class, 'classPrior, 'evidence)
      .groupBy('id, 'class) {
        _.sum('evidence -> 'sumEvidence)
         .max('classPrior)
      }
    ret
  }

  private def _gaussian_prob(theta : Double, sigma : Double, score : Double) : Double = {
    // from sklearn:
    // n_ij = - 0.5 * np.sum(np.log(np.pi * self.sigma_[i, :]))
    //     n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /
    //                          (self.sigma_[i, :]), 1)
    // val (theta, sigma, score) = values
    val outside = -0.5 * math.log(math.Pi * sigma)
    val expo = 0.5 * math.pow(score - theta, 2) / sigma
    outside - expo
  }

  /**
  * Calculates the size, mean, and standard deviation for each class/feature
  * pair.
  */
  private def featureStats(pipe : Pipe, nReducers : Int = 50)(implicit fd: FlowDef) : Pipe = {
    pipe
      .groupBy('feature, 'class) {
        _.sizeAveStdev('score -> ('featureClassSize, 'theta, 'sigma))
         .reducers(nReducers)
      }
  }

}



/** BaseDiscreteNB is a classifier for features with discrete features, such
  * as things like "month of registration" or "eye color". The training input
  * does not use a `score` field, just the feature's enumerated value.
  */
trait BaseDiscreteNB extends NBCore {
  import Dsl._
  /** Train a discrete Naive Bayes model.
    *
    * Output model contains the fields: `classId`, `feature`, `evidence`
    */
  def train(pipe : Pipe, nReducers : Int = 100)(implicit fd: FlowDef) : Pipe = {

    val prClass = classPrior(pipe, nReducers)
    val prFeatureClass = featureStats(pipe, nReducers)

    val model = pipe
      .joinWithSmaller('class -> 'class, prClass, reducers=nReducers)
      .joinWithSmaller(('class, 'feature) -> ('class, 'feature), prFeatureClass, reducers=nReducers)
      .mapTo(('class, 'classPrior, 'classCount, 'feature, 'featureClassSize) ->
             ('class, 'feature, 'evidence, 'classPrior)) {
        values : (String, Double, Double, String, Double) =>
        val (classId, classPrior, classCount, feature, featureClassSize) = values
        val featureLogProb = math.log(featureClassSize / classCount)
        (classId, feature, featureLogProb, classPrior)
      }

    model
  }

  def _joint_log_likelihood(joined : Pipe)(implicit fd: FlowDef) : Pipe = {
    val res = joined
      .groupBy('id, 'class) {
        _.sum('evidence -> 'sumEvidence)
         .max('classPrior)
      }
    res
  }

  def featureStats(pipe : Pipe, nReducers : Int = 50)(implicit fd: FlowDef) : Pipe = pipe.groupBy('feature, 'class) {_.size('featureClassSize).reducers(nReducers)}
}


/**
  * MultinomialNB should be used for classification for data with discrete
  * features, such as word counts.
  *
  * The training data should have three fields: `class`, `feature` and `score`.
  */
object MultinomialNB extends BaseDiscreteNB {
  import Dsl._
  /** Overwrite classPrior to calculate `sum(score | class) / sum(score)`
    * instead of `count(class) / count(all rows)`
    */
  override def classPrior(pipe : Pipe, nReducers : Int = 50)(implicit fd: FlowDef) : Pipe = {
    val counts = pipe.groupBy('class) { _.sum('score -> 'classCount).reducers(nReducers) }
    val totSum = counts.groupAll(_.sum('classCount -> 'totalCount))

    counts.crossWithTiny(totSum)
      .mapTo(('class, 'classCount, 'totalCount) -> ('class, 'classPrior, 'classCount)) {
        x : (String, Double, Double) => (x._1, math.log(x._2 / x._3), x._2)
      }
  }
  override def featureStats(pipe : Pipe, nReducers : Int = 50)(implicit fd: FlowDef) : Pipe = {
    pipe.groupBy('feature, 'class) {_.sum('score -> 'featureClassSize).reducers(nReducers)}
  }

  override def _joint_log_likelihood(joined : Pipe)(implicit fd: FlowDef) : Pipe = {
    val res = joined
      .map(('score, 'evidence) -> 'sumEvidence) {
        values : (Double, Double) => values._1 * values._2
      }
    res
  }
}
