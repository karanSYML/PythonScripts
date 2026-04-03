public static void main(String[] args) throws PatriusException, IOException{
        // Scenario : 
        // We have a satellite with TLE format (two line element)
        // We have a ground station defined
        // The goal is to see when the satellite will be visible by the ground station 
        // ----------------------------------------------------------------------------//
    

        // Patrius Dataset initialization (needed for example to get the UTC time)
        PatriusDataset.addResourcesFromPatriusDataset() ;

        // Recovery of the UTC time scale using a "factory" (not to duplicate such unique object)
        final TimeScale TUC = TimeScalesFactory.getUTC();
        
        // TLE to propagate
        String line1ISS = "1 22796U 93058B   26085.24219724 -.00000118  00000-0  00000-0 0  9994";
        String line2ISS = "2 22796  14.5776 356.2114 0052450  77.3744  91.3947  1.00272108126078";
        final TLE ISS_TLE = new TLE(line1ISS, line2ISS);

        // Start date of the propagation : date of the TLE
        final AbsoluteDate startDate = ISS_TLE.getDate();

        //Attitude proivider 
        final AttitudeProvider law = new ConstantAttitudeLaw(FramesFactory.getTEME(), Rotation.IDENTITY);

        // propagator
        final TLEPropagator propagator = TLEPropagator.selectExtrapolator(ISS_TLE, law, null);

        // the initial state of the propagator
        final SpacecraftState initialState = propagator.getInitialState();
        
        // Array of two elements 
        // start and end of the visibility event
        AbsoluteDate visbilityDate[] = new AbsoluteDate[2];


        //The ground station defined by a name, longitude, latitude, halp opening angle
        GroundStation station = GroundStation.initFrameAndField("CLEVELAND NGS", FastMath.toRadians(-116.874), FastMath.toRadians(35.342), FastMath.PI / 12.);

        //define our satellite, by its related ground station, initial state of the propagator of its orbit, and frequency min and frequency max
        IOSatellite testSat = new IOSatellite(station, initialState, 0., 0.);
        System.out.println("initial state is either : "+ initialState+ " or "+testSat.getState()+ " and then "+ initialState.getAttitude());
        
        // some other parameters 
        final double maxCheck = 10.;
        final double threshold = 10.e-10;
        
        // detector of mutual visibility between a satellite sensor and a ground station sensor supposed to be ponctual
        // We don't use a correction model  
        final EventDetector detectorMainPart = new StationToSatMutualVisibilityDetector(testSat.getSensor(), station, null, false, maxCheck, threshold);
        
        // Use the precedent detector to get the start date as it will be the first occurence and the and date as itr is not the first occurence
        final NthOccurrenceDetector firstSatVisibility = new NthOccurrenceDetector(detectorMainPart, 0, Action.CONTINUE) {
            @Override
            public Action eventOccurred(SpacecraftState s, boolean increasing, boolean forward) throws PatriusException {
                super.eventOccurred(s, increasing, forward);
                if (getCurrentOccurrence() == getOccurence()) {
                    visbilityDate[0] = s.getDate();
                    System.out.println("first date to detect the srat of event is "+s.getDate().toString(TUC));
                } else {
                    visbilityDate[1] = s.getDate();
                    System.out.println("last date to detect the srat of event is "+s.getDate().toString(TUC));
                }
                return Action.CONTINUE;
            }
        };
        // Propagation
        System.out.println(propagator.getInitialState().getAttitude());
        propagator.addEventDetector(firstSatVisibility);
        // Simulate a full period of a GPS satellite
        // -----------------------------------------
        final SpacecraftState finalState = propagator.propagate(startDate.shiftedBy(8600000));

        // Check results
        System.out.println(initialState.getA()+" and " + finalState.getA());
        System.out.println(initialState.getEquinoctialEx()+" and " +finalState.getEquinoctialEx());
        System.out.println(initialState.getEquinoctialEy()+" and " + finalState.getEquinoctialEy());
        System.out.println(initialState.getHx()+" and " + finalState.getHx());
        System.out.println(initialState.getHy()+" and " + finalState.getHy());
        System.out.println(initialState.getLM()+" and " + finalState.getLM());
        System.out.println("date required is :"+ visbilityDate[1]);
        SpringApplication.run(DemoApplication.class, args); 
    }