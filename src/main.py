import kalman_filter as kalman_filter
import yahoo_financedata as yahoo_financedata
import numpy as np
import matplotlib.pyplot as plt


def main():

    # read data
    data_filename = 'data/IFNNY.csv'
    dataReader = yahoo_financedata.YahooFinanceData()
    dataReader.open_data(data_filename)

    # init kalman filter
    x = np.full((2, 1), 2)
    P = np.full((2, 2), 2 ** 2)
    state = kalman_filter.State(x, P)

    Q = np.full((2, 2), 1 ** 2)
    F = np.array([[1, 1],
                  [0, 1]])
    R = np.array([[0.5 ** 2]])
    H = np.array([[1, 0]])

    # data for plotting
    gt = {'price': [], 'trend': []}
    result = {'price': [], 'trend': []}
    velocity = []

    counter = 0

    while dataReader.has_more_data():
        z = dataReader.next_measurement()

        if counter == 0:
            x = np.array([[z[0][0]], [0]])
            state = kalman_filter.State(x, P)

        prior = kalman_filter.predict(state, F, Q)
        posterior = kalman_filter.update(prior, z, R, H)
        state = posterior

        gt['price'].append(z[0][0])
        result['price'].append(prior.X[0][0])
        velocity.append(prior.X[1][0])

        if counter == 0:
            gt['trend'].append(1)
            result['trend'].append(1)
        else:
            predicted_trend = 1 if prior.X[0][0] > result['price'][counter - 1] else -1
            result['trend'].append(predicted_trend)
            gt_trend = 1 if z[0][0] > gt['price'][counter - 1] else -1
            gt['trend'].append(gt_trend)

        counter += 1
        # if counter > 20:
        #    break

    # KPI
    # number of correct trend predictions
    correct_predictions = 0
    for x, y in zip(result['trend'], gt['trend']):
        if x == y:
            correct_predictions += 1
    # print(correct_predictions)
    # print(len(gt['trend']))


    price_mse = np.mean(((np.array(gt['price']) - np.array(result['price']))**2 ))
    print("Stock Price MSE: {}".format(price_mse))
    print( "Trend Accuarcy: {}".format( float(correct_predictions) / len(gt['trend'])))

    plt.figure(1)

    plt.plot(np.array(gt['price']), linewidth=0.7)
    plt.plot(np.array(result['price'] )+ np.random.normal(-1, 0.5, len(result['price'])), linewidth=0.7)

    plt.legend(['actual stock price', 'predicted stock price'])
    plt.show()


    plt.plot(velocity)
    plt.legend(['velocity'])
    plt.show()

    x_axis = list(range(len(gt['trend'])))
    plt.scatter(x_axis, gt['trend'], color='g', alpha=0.5)
    plt.scatter(x_axis, result['trend'], marker='x', color='r', alpha=0.5)

    plt.legend(['actual trend', 'predicted trend'])

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
