async user_attendee_list_new(req, res) {
  try {
    const userId = new mongoose.Types.ObjectId(req.user_id);
    const start = parseInt(req.query.start) || 0;
    const limit = parseInt(req.query.limit) || 50;
    const sortDir = req.query.sort_by === "des" ? -1 : 1;
    const isRecommended = req.query.is_recommended === "true";

    // Base search
    let search_obj = {
      is_deleted: false,
      is_email_verified: true,
      role: "user",
      _id: { $ne: userId },
    };

    if (req.query.event_id) {
      const eventIds = req.query.event_id.split(",").map(id => new mongoose.Types.ObjectId(id));
      const registeredUserIds = await event_registrations.distinct("user_id", {
        event_id: { $in: eventIds },
        user_id: { $ne: userId },
        is_deleted: false,
      });
      search_obj._id = { $in: registeredUserIds };
    }

    let profile_completed = false;
    let delegate_pass = false;
    const loginInfo = await login_info.findOne({ _id: userId }).lean();
    if (loginInfo) profile_completed = loginInfo.profile_percentage === 100;
    const pass = await attendee_passes.findOne({ email: loginInfo?.email_id }).sort({ _id: -1 }).lean();
    delegate_pass = pass && ["TNGSS Conference", "Conference"].includes(pass?.pass_type);

    // === STEP 1: Fetch AI scores if recommended ===
    let aiScoreMap = {};
    if (isRecommended) {
      const aiScores = await mongoose.connection
        .collection("user_ai_score")
        .find({ user_id: userId }) // userId must be ObjectId
        .project({ matched_user_id: 1, score: 1 })
        .sort({ score: -1 }) // DB-side high → low
        .toArray();

      aiScores.forEach(s => {
        aiScoreMap[s.matched_user_id.toString()] = s.score;
      });
    }

    let pipeline = [
      { $match: search_obj },
      { $addFields: { sort_key: { $cond: [{ $eq: ["$first_name", ""] }, 1, 0] } } },

      {
        $lookup: {
          from: "attendee-passes",
          let: { email: "$email_id" },
          pipeline: [
            { $match: { $expr: { $eq: ["$email", "$$email"] } } },
            ...(req.query.organisation_type ? [{ $match: { organisation_type: req.query.organisation_type } }] : []),
            ...(req.query.sector_interested ? [{ $match: { sector_interested: req.query.sector_interested } }] : []),
            { $project: { organisation_type: 1, sector_interested: 1 } },
            { $limit: 1 },
          ],
          as: "attendee_passes_details",
        },
      },

      {
        $lookup: {
          from: "profile_management",
          let: { userId: "$_id" },
          pipeline: [
            { $match: { $expr: { $eq: ["$user_id", "$$userId"] } } },
            ...(req.query.profile_type ? [{ $match: { profile_type: req.query.profile_type } }] : []),
            ...(req.query.focused_sector ? [{ $match: { focused_sector: req.query.focused_sector } }] : []),
            { $project: { designation: 1, organization_name: 1, bio: 1, profile_type: 1, focused_sector: 1 } },
            { $limit: 1 },
          ],
          as: "profile_details",
        },
      },

      {
        $lookup: {
          from: "connection_request_management",
          let: { userId: "$_id", currentUserId: userId },
          pipeline: [
            {
              $match: {
                $expr: {
                  $and: [
                    { $or: [{ $eq: ["$request_from", "$$currentUserId"] }, { $eq: ["$request_to", "$$currentUserId"] }] },
                    { $or: [{ $eq: ["$request_from", "$$userId"] }, { $eq: ["$request_to", "$$userId"] }] },
                    { $ne: ["$status", "declined"] },
                    { $eq: ["$is_deleted", false] },
                  ],
                },
              },
            },
            { $sort: { _id: -1 } },
            { $limit: 1 },
          ],
          as: "connection_info",
        },
      },

      ...(isRecommended ? [] : [{ $sort: { sort_key: 1, first_name: sortDir } }]),
    ];

    pipeline.push({
      $project: {
        user_id: "$_id",
        first_name: { $ifNull: ["$first_name", ""] },
        profile_image: 1,
        designation: { $ifNull: [{ $arrayElemAt: ["$profile_details.designation", 0] }, ""] },
        organization_name: { $ifNull: [{ $arrayElemAt: ["$profile_details.organization_name", 0] }, ""] },
        profile_percentage: { $ifNull: ["$profile_percentage", 0] },
        bio_exists: { $ne: [{ $ifNull: [{ $arrayElemAt: ["$profile_details.bio", 0] }, ""] }, ""] },
        already_connection_request_send: { $ne: [{ $arrayElemAt: ["$connection_info._id", 0] }, null] },
        connection_request_status: { $ifNull: [{ $arrayElemAt: ["$connection_info.status", 0] }, ""] },
        connection_reference_id: { $ifNull: [{ $arrayElemAt: ["$connection_info._id", 0] }, ""] },
        sender_or_reciever: {
          $switch: {
            branches: [
              { case: { $eq: [{ $toString: { $arrayElemAt: ["$connection_info.request_from", 0] } }, req.user_id] }, then: "sender" },
              { case: { $ne: [{ $arrayElemAt: ["$connection_info.request_from", 0] }, null] }, then: "receiver" },
            ],
            default: "",
          },
        },
        organisation_type: { $arrayElemAt: ["$attendee_passes_details.organisation_type", 0] },
        sector: { $arrayElemAt: ["$attendee_passes_details.sector_interested", 0] },
        profile_type: { $arrayElemAt: ["$profile_details.profile_type", 0] },
        focused_sector: { $arrayElemAt: ["$profile_details.focused_sector", 0] },
        ai_enabled: { $and: [delegate_pass, profile_completed, { $eq: ["$profile_percentage", 100] }] },
      },
    });

    if (start > 0) pipeline.push({ $skip: start });
    if (limit > 0) pipeline.push({ $limit: limit });

    const attendees = await login_info.aggregate(pipeline).allowDiskUse(true);
    const total_count = await login_info.countDocuments(search_obj);

    const finalList = attendees.map(a => ({
      ...a,
      ai_score: aiScoreMap[a.user_id.toString()] || 0,
    }));

    if (isRecommended) {
      finalList.sort((a, b) => b.ai_score - a.ai_score); // High → Low
    }

    return jsend(200, "Successfully retrieved the attendees list", {
      total_count,
      attendee_list: finalList,
    });
  } catch (e) {
    return jsend(406, e.message);
  }
}
