from builders.model_builder import ModelBuilder
from decorators.class_decorators import ClassDecorators

class ModelBuilderDecorators:

    @ClassDecorators.add_to_class(ModelBuilder)
    def logistic_regression(self, X_train, X_test, y_train, y_test, *args):
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        self.logger.info("Logistic Regression test set accuracy: {}".format(
            str(model.score(X_test, y_test))
        ))
        return model

    @ClassDecorators.add_to_class(ModelBuilder)
    def sgd_classifier(self, X_train, X_test, y_train, y_test, *args):
        from sklearn.linear_model import SGDClassifier

        model = SGDClassifier(max_iter=500)
        model.fit(X_train, y_train)

        self.logger.info("SGD Classifier test set accuracy: {}".format(
            str(model.score(X_test, y_test))
        ))
        return model

    @ClassDecorators.add_to_class(ModelBuilder)
    def knn_classifier(self, X_train, X_test, y_train, y_test, *args):
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)

        self.logger.info("KNN Classifier test set accuracy: {}".format(
            str(model.score(X_test, y_test))
        ))
        return model