
include "utils.lua"

include "MaskedLoss.lua"

include 'buildData.lua'
include "nl2code_extension.lua"


-- Server = {}
function get_predictions(xbatch, ybatch, enc, dec)
	local ps = {}
	for d = 1, 2 * params.layers do
		ps[d] = torch.zeros(1, params.rnn_size):cuda() -- for prediction
	end
	
	xbatch.x = xbatch.x:expand(params.max_code_length, params.batch_size)
	xbatch.mask = xbatch.mask:expand(params.max_code_length, params.batch_size)
	xbatch.fmask = xbatch.fmask:expand(params.max_code_length, params.batch_size)
	xbatch.infmask = xbatch.infmask:expand(params.max_code_length, params.batch_size)
	xbatch.xsizes = xbatch.xsizes:expand(params.batch_size)

	local all_h = enc:forward(xbatch.x)

	return computeProb(ybatch, ps, all_h:narrow(1, 1, 1), xbatch.infmask:narrow(2, 1, 1), dec)

end

function computeProb(batch, prevs, all_h, infmask, dec)
	local y = torch.ones(1):cuda()

	local i = 1
	local prob = 0

	
	repeat


		local tmp = dec:forward({batch.y[i], y, torch.ones(1):cuda() * i, prevs, all_h, infmask})[2]
		
		local fnodes = dec.forwardnodes
		local pred_vector = fnodes[#fnodes].data.mapindex[1].input[1][1]
		
		prob = prob + pred_vector[batch.y[i + 1][1]]
		
		copy_table(prevs, tmp)
		i = i + 1
	until i > params.max_nl_length or batch.y[i][1] == 1

	return prob
end

function dist_compare(a, b)
	return a.prob > b.prob
end



function succesrate(real, predict, K) 
	local succesrate = 0
	
	for i = 1, K do
		
		if predict[i].i[1] == tonumber(real) then
			succesrate = 1
		end 
	end
	return succesrate
end

function MRR(real, predict) 
	local MRR = 0
	for i = 1, 10 do
		if predict[i].i[1] == tonumber(real) then
			MRR = 1 / (i + 1)
		end 
	end
	return MRR
end

function main()

	local cmd = torch.CmdLine()
	cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
	cmd:option('-encoder',  'None', 'Previously trained encoder')
	cmd:option('-decoder',  'None', 'Previously trained decoder')
	cmd:option('-rnn_size', 400, 'Dimension')
	cmd:option('-language', 'code', 'Code language')
	cmd:option('-port', 9090, 'Server port')
	cmd:option('-max_nl_length', 80, 'length')
	cmd:option('-max_code_length', 80, 'length')

	local working_dir = os.getenv("CODENN_WORK")

	cmd:text()
	opt = cmd:parse(arg)

	params =      {
		max_length=80,
		layers=1,
		max_code_length=opt.max_code_length,
		max_nl_length=opt.max_nl_length,
		batch_size=20,
	  rnn_size=opt.rnn_size
	}

	init_gpu(opt.gpuidx)


	--preload vocab and models
	vocab = torch.load(working_dir .. '/vocab.data.' .. opt.language)
	print(type(vocab))
	print("Vocabulary loaded")
  	encoderCell = torch.load(opt.encoder)
  	decoderCell= torch.load(opt.decoder)

	state_train = torch.load(working_dir .. '/dev.data.' .. opt.language)
	state_train.name = "training"


	print("Models loaded")

	g_disable_dropout(encoderCell)
	g_disable_dropout(decoderCell)
	local state = 0

	scores = {}	
	sc1 = {}	
	sc5 = {}
	sc10 = {}
	MRR_scores = {}
	local j = 0
	for line in io.lines(working_dir .. '/ref.txt') do

		local query_info={}
		for item in string.gmatch(line, "([^"..'\t'.."]+)") do
			table.insert(query_info, item)
		 end

		local filename = '/tmp/nl2code.tmpfile'
		local f = io.open(filename, 'w')
		f:write('23\t23\t' .. tostring(query_info[3]) .. '\t' .. 'function()' .. '\t' .. '0.45')
		f:close()


        local data = get_data_map(filename, vocab, true, params.max_code_length, params.max_nl_length)
		local file_name_2 = (((os.getenv("CODENN_WORK") .. "/") .. 'nl2code.tmpfile') .. ".") .. opt.language
		local data = get_data(file_name_2, vocab, 1, params.max_code_length, params.max_nl_length)
		local n_batch = getInCuda(data.batches[1])
		
		local ranks = {}

			-- Now we need to convert 
		for i = 1, #state_train.batches do
			local batch = state_train.batches[i]
			local c_batch = getInCuda(batch)
			prob = get_predictions(c_batch, n_batch, encoderCell, decoderCell)
			table.insert(ranks, {p=prob, c=batch.code, i=batch.ids})
			
		end

	    table.sort(ranks, function (a,b) 
	    	return a.p > b.p
			end)

		sc1[j] = succesrate(query_info[2], ranks, 1)
		sc5[j] = succesrate(query_info[2], ranks, 5)
		sc10[j] = succesrate(query_info[2], ranks, 10)
		MRR_scores[j]= MRR(query_info[2], ranks)

		state = state + 1
		j = j + 1  


		if math.fmod(j, 10) == 0 then
			scores['sc1'] = sc1
			scores['sc5'] = sc5
			scores['sc10'] = sc10
			scores['MRR'] = MRR_scores
			local file_scores = assert(io.open(working_dir .. "/scores." .. 'json', 'w'))
			local all_scores = JSON:encode(scores)
			file_scores:write(all_scores)
			file_scores:close()
		end
	end
	file:close()
end



if script_path() == "nl2code_org.lua" then
	main()
end