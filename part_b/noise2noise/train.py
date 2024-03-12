from imports import *

DEVICE = torch.device('cuda')


def train(net, data_object, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):

    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        # first evaluating for comparison

        if e == 0 and data_object.training_type == "Noise2Clean":
            print("Pre-training evaluation")
            # with torch.no_grad():
            #    test_loss,testmet = test_epoch(net, test_loader, loss_fn,use_net=False)
            # print("Had to load model.. checking if deets match")
            # again, modified cuz im loading
            testmet = getMetricsonLoader(test_loader, net, False)
            # test_losses.append(test_loss)
            # print("Loss before training:{:.6f}".format(test_loss))

            with open(data_object.basepath + "/results.txt", "w+") as f:
                f.write("Initial : \n")
                f.write(str(testmet))
                f.write("\n")

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        test_loss = 0
        scheduler.step()
        print("Saving model....")

        with torch.no_grad():
            test_loss, testmet = test_epoch(
                net, test_loader, loss_fn, use_net=True)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # print("skipping testing cuz peak autism idk")

        with open(data_object.basepath + "/results.txt", "a") as f:
            f.write("Epoch :"+str(e+1) + "\n" + str(testmet))
            f.write("\n")

        print("OPed to txt")

        torch.save(net.state_dict(), data_object.basepath +
                   '/Weights/dc20_model_'+str(e+1)+'.pth')
        torch.save(optimizer.state_dict(), data_object.basepath +
                   '/Weights/dc20_opt_'+str(e+1)+'.pth')

        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # print("Epoch: {}/{}...".format(e+1, epochs),
        #              "Loss: {:.6f}...".format(train_loss),
        #              "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss


def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0
    for noisy_x, clean_x in train_loader:

        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)

        # zero  gradients
        net.zero_grad()

        # get the output from the model
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss


def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0.
    counter = 0.
    '''
    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        test_ep_loss += loss.item() 
        
        counter += 1

    test_ep_loss /= counter
    '''

    # print("Actual compute done...testing now")

    testmet = getMetricsonLoader(test_loader, net, use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return test_ep_loss, testmet
    # test_ep_loss = test_loss
    # testmet = testmet
